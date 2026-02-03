#=
Distributed Training Script for NeuralMSE Models

This script trains multiple NBE and NPE models in parallel using Julia's
distributed computing via SlurmClusterManager. Each worker saves to its own
file to avoid concurrent file access, then a single worker collects all
results at the end.

Usage:
    # Submit as SLURM job
    sbatch scripts/train_job.sh

    # Run locally with --local flag
    julia --project scripts/train_models_distributed.jl --local
=#

using Distributed, SlurmClusterManager

# Parse command line arguments
const USE_LOCAL = "--local" in ARGS

if USE_LOCAL
    # Local execution - use available cores
    n_workers = max(1, Sys.CPU_THREADS ÷ 2 - 1)
    println("Running locally with $n_workers workers")
    addprocs(n_workers; exeflags="--project=$(Base.active_project())")
else
    # SLURM execution
    if !haskey(ENV, "SLURM_JOB_ID")
        error("Not running under SLURM. Use --local flag for local execution, or submit via sbatch.")
    end

    using SlurmClusterManager

    job_id = ENV["SLURM_JOB_ID"]
    println("SLURM Job ID: $job_id")

    # Add workers via SlurmClusterManager
    addprocs(SlurmManager(); exeflags=["--project=$(Base.active_project())"])
    println("Added $(nworkers()) workers")
end

println("Running with $(nworkers()) workers")

# Load packages on all workers
@everywhere begin
    using NeuralMSE
    using NeuralMSE: ModelConfig
    using Dates
    using JLD2
end

#=
Worker function: Train and save to individual file (no concurrent access)
=#

@everywhere begin
    """
    Train a model and save to an individual temp file.
    Returns the path to the saved file, or nothing if skipped/failed.
    """
    function train_and_save_individual(
        temp_dir::String,
        job_idx::Int,
        model_type::Symbol,
        K::Int;
        censoring_lower::Int=0,
        censoring_upper::Int=0,
        width::Int=256,
        n_hidden::Int=3,
        encoding_dim::Int=128,
        train_size::Int=10000,
        epochs_nbe::Int=100,
        epochs_npe::Int=200,
        m::Int=1,
        verbose::Bool=true
    )
        worker_id = myid()
        hostname = gethostname()

        # Each job gets its own output file
        output_file = joinpath(temp_dir, "job_$(job_idx).jld2")

        verbose && println("[Worker $worker_id@$hostname] Training $model_type K=$K censor=($censoring_lower,$censoring_upper)...")
        train_start = now()

        try
            if model_type == :nbe
                point_est, ci_est = train_nbe(K;
                    width=width,
                    n_hidden=n_hidden,
                    train_size=train_size,
                    m=m,
                    epochs=epochs_nbe,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper,
                    savepath=nothing,
                    verbose=verbose
                )

                config = ModelConfig(
                    model_type=:nbe,
                    K=K,
                    width=width,
                    n_hidden=n_hidden,
                    train_size=train_size,
                    m=m,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper
                )

                # Save to individual file
                jldsave(output_file;
                    model_type=:nbe,
                    point_estimator=point_est,
                    interval_estimator=ci_est,
                    config=config
                )

            elseif model_type == :npe
                estimator = train_npe(K;
                    width=width,
                    n_hidden=n_hidden,
                    encoding_dim=encoding_dim,
                    train_size=train_size,
                    m=m,
                    epochs=epochs_npe,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper,
                    savepath=nothing,
                    verbose=verbose
                )

                config = ModelConfig(
                    model_type=:npe,
                    K=K,
                    width=width,
                    n_hidden=n_hidden,
                    encoding_dim=encoding_dim,
                    train_size=train_size,
                    m=m,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper
                )

                # Save to individual file
                jldsave(output_file;
                    model_type=:npe,
                    estimator=estimator,
                    config=config
                )
            else
                error("Unknown model type: $model_type")
            end

            train_elapsed = now() - train_start
            verbose && println("[Worker $worker_id@$hostname] Completed in $train_elapsed, saved to $output_file")

            return output_file

        catch e
            verbose && println("[Worker $worker_id@$hostname] ERROR: $(sprint(showerror, e))")
            rethrow(e)
        end
    end
end

#=
Main process: Collect all individual files into final registry
=#

"""
Collect all trained models from temp files into the final models directory.
This runs on the main process only, avoiding concurrent file access.
"""
function collect_models(temp_dir::String, models_dir::String; verbose::Bool=true)
    verbose && println("\nCollecting trained models...")

    mkpath(models_dir)

    # Find all job files
    job_files = filter(f -> startswith(f, "job_") && endswith(f, ".jld2"), readdir(temp_dir))

    if isempty(job_files)
        verbose && println("No trained models found in $temp_dir")
        return 0
    end

    verbose && println("Found $(length(job_files)) trained models to collect")

    collected = 0
    for job_file in job_files
        filepath = joinpath(temp_dir, job_file)
        try
            jldopen(filepath, "r") do f
                model_type = f["model_type"]
                config = f["config"]

                # Check if model already exists
                if model_exists(models_dir; K=config.K, model_type=config.model_type,
                               censoring_lower=config.censoring_lower,
                               censoring_upper=config.censoring_upper)
                    verbose && println("  Skipping (already exists): $(config.model_type) K=$(config.K)")
                    return
                end

                if model_type == :nbe
                    point_est = f["point_estimator"]
                    interval_est = f["interval_estimator"]
                    model_id = save_nbe_model(models_dir, point_est, interval_est, config)
                    verbose && println("  Collected NBE model ID=$model_id: K=$(config.K) censor=($(config.censoring_lower),$(config.censoring_upper))")
                else
                    estimator = f["estimator"]
                    model_id = save_model(models_dir, estimator, config)
                    verbose && println("  Collected NPE model ID=$model_id: K=$(config.K) censor=($(config.censoring_lower),$(config.censoring_upper))")
                end
                collected += 1
            end
        catch e
            verbose && println("  Error processing $job_file: $(sprint(showerror, e))")
        end
    end

    verbose && println("Collected $collected models")
    return collected
end

#=
Training configuration and execution
=#

"""
Generate list of training jobs.
"""
function generate_training_jobs(;
    K_values::Vector{Int}=[3, 4, 5, 6, 7],
    model_types::Vector{Symbol}=[:nbe, :npe],
    censoring_configs::Vector{Tuple{Int,Int}}=[(0, 0)]
)
    jobs = []
    for K in K_values
        for model_type in model_types
            for (censoring_lower, censoring_upper) in censoring_configs
                push!(jobs, (
                    model_type=model_type,
                    K=K,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper
                ))
            end
        end
    end
    return jobs
end

"""
Run distributed training for all jobs.
"""
function train_all_models(
    models_dir::String;
    K_values::Vector{Int}=[3, 4, 5, 6, 7],
    model_types::Vector{Symbol}=[:nbe, :npe],
    censoring_configs::Vector{Tuple{Int,Int}}=[(0, 0)],
    width::Int=256,
    n_hidden::Int=3,
    encoding_dim::Int=128,
    train_size::Int=10000,
    epochs_nbe::Int=100,
    epochs_npe::Int=200,
    m::Int=1,
    verbose::Bool=true
)
    # Create temp directory for individual job outputs
    temp_dir = joinpath(models_dir, ".temp_training")
    mkpath(temp_dir)

    jobs = generate_training_jobs(;
        K_values=K_values,
        model_types=model_types,
        censoring_configs=censoring_configs
    )

    println("=" ^ 70)
    println("Distributed Training Configuration")
    println("=" ^ 70)
    println("Output directory: $models_dir")
    println("Temp directory: $temp_dir")
    println("Workers: $(nworkers())")
    println("Total jobs: $(length(jobs))")
    println("K values: $K_values")
    println("Model types: $model_types")
    println("Censoring configs: $(length(censoring_configs)) configurations")
    println("Architecture: width=$width, n_hidden=$n_hidden, encoding_dim=$encoding_dim")
    println("Training: train_size=$train_size, epochs_nbe=$epochs_nbe, epochs_npe=$epochs_npe, m=$m")
    if haskey(ENV, "SLURM_JOB_ID")
        println("SLURM Job: $(ENV["SLURM_JOB_ID"])")
        println("SLURM Nodes: $(get(ENV, "SLURM_JOB_NODELIST", "unknown"))")
    end
    println("=" ^ 70)
    println()

    start_time = now()

    # Phase 1: Train models in parallel, each saving to its own file
    println("Phase 1: Training models (each worker saves to individual file)...")

    results = pmap(enumerate(jobs); on_error=ex -> ex) do (idx, job)
        try
            output_file = train_and_save_individual(
                temp_dir,
                idx,
                job.model_type,
                job.K;
                censoring_lower=job.censoring_lower,
                censoring_upper=job.censoring_upper,
                width=width,
                n_hidden=n_hidden,
                encoding_dim=encoding_dim,
                train_size=train_size,
                epochs_nbe=epochs_nbe,
                epochs_npe=epochs_npe,
                m=m,
                verbose=verbose
            )
            return (job=job, output_file=output_file, error=nothing)
        catch e
            return (job=job, output_file=nothing, error=sprint(showerror, e, catch_backtrace()))
        end
    end

    training_elapsed = now() - start_time
    println("\nPhase 1 complete. Training time: $training_elapsed")

    # Phase 2: Collect all results on main process (no concurrent access)
    println("\nPhase 2: Collecting results (single process)...")
    collect_start = now()

    collected = collect_models(temp_dir, models_dir; verbose=verbose)

    collect_elapsed = now() - collect_start
    println("Phase 2 complete. Collection time: $collect_elapsed")

    # Summary
    total_elapsed = now() - start_time

    println()
    println("=" ^ 70)
    println("Training Complete")
    println("=" ^ 70)
    println("Total elapsed time: $total_elapsed")
    println("  Training: $training_elapsed")
    println("  Collection: $collect_elapsed")

    # Count results
    processed_results = map(results) do r
        if r isa Exception
            return (job=nothing, output_file=nothing, error=sprint(showerror, r))
        else
            return r
        end
    end

    successes = filter(r -> r.error === nothing && r.output_file !== nothing, processed_results)
    failures = filter(r -> r.error !== nothing, processed_results)

    println("\nTraining results:")
    println("  Successful: $(length(successes))")
    println("  Failed: $(length(failures))")
    println("  Collected to registry: $collected")

    if !isempty(failures)
        println("\nFailures:")
        for r in failures
            job_desc = r.job !== nothing ? "$(r.job.model_type) K=$(r.job.K)" : "unknown job"
            println("  $job_desc:")
            error_lines = split(string(r.error), '\n')
            for line in error_lines[1:min(3, length(error_lines))]
                println("    $line")
            end
        end
    end

    # Show final model count
    println("\nModels in registry:")
    try
        df = list_models(models_dir)
        println("  Total: $(nrow(df))")
        for mt in unique(df.model_type)
            count = sum(df.model_type .== mt)
            println("  $mt: $count")
        end
    catch e
        println("  (Could not read registry: $e)")
    end

    println("=" ^ 70)

    # Optionally clean up temp directory
    # rm(temp_dir; recursive=true, force=true)

    return results
end

#=
Main execution
=#

# Default training configuration - modify these as needed
const MODELS_DIR = get(ENV, "NEURALMSE_MODELS_PATH",
                       joinpath(dirname(dirname(@__FILE__)), "trained_models"))

const K_VALUES = collect(3:15)
const MODEL_TYPES = [:nbe, :npe]
const CENSORING_CONFIGS = [(x, y) for x in 0:1 for y in x:16]

# Training hyperparameters
const WIDTH = 256
const N_HIDDEN = 3
const ENCODING_DIM = 128
const TRAIN_SIZE = 10000
const EPOCHS_NBE = 100
const EPOCHS_NPE = 200
const M = 1

# Run training
train_all_models(
    MODELS_DIR;
    K_values=K_VALUES,
    model_types=MODEL_TYPES,
    censoring_configs=CENSORING_CONFIGS,
    width=WIDTH,
    n_hidden=N_HIDDEN,
    encoding_dim=ENCODING_DIM,
    train_size=TRAIN_SIZE,
    epochs_nbe=EPOCHS_NBE,
    epochs_npe=EPOCHS_NPE,
    m=M,
    verbose=true
)
