#=
Distributed Training Script for NeuralMSE Models

This script trains multiple NBE and NPE models in parallel using Julia's
distributed computing capabilities. Supports both local execution and
SLURM cluster environments via SlurmClusterManager.

Usage:
    # Submit as SLURM job (recommended)
    sbatch scripts/train_job.sh

    # Or run locally for testing
    julia --project scripts/train_models_distributed.jl --local
=#

using Distributed

# Parse command line arguments
const USE_LOCAL = "--local" in ARGS

if USE_LOCAL
    # Local execution - use available cores
    n_workers = max(1, Sys.CPU_THREADS ÷ 2 - 1)
    println("Running locally with $n_workers workers")
    addprocs(n_workers; exeflags="--project")
else
    # SLURM execution
    if !haskey(ENV, "SLURM_JOB_ID")
        error("Not running under SLURM. Use --local flag for local execution, or submit via sbatch.")
    end

    using SlurmClusterManager

    # Get SLURM allocation info
    n_tasks = parse(Int, get(ENV, "SLURM_NTASKS", "1"))
    job_id = ENV["SLURM_JOB_ID"]

    println("SLURM Job ID: $job_id")
    println("Requested tasks: $n_tasks")

    # Add workers via SlurmClusterManager
    # This spawns workers across the allocated nodes/tasks
    addprocs(SlurmManager(); exeflags=["--project=$(Base.active_project())"])

    println("Added $(nworkers()) workers")
end

println("Running with $(nworkers()) workers on $(length(unique(map(w -> Distributed.remotecall_fetch(() -> gethostname(), w), workers())))) node(s)")

# Load packages on all workers
@everywhere begin
    using NeuralMSE
    using Dates
end

#=
File-based locking for safe concurrent registry updates
=#

@everywhere begin
    """
    Acquire an exclusive lock on the model registry.
    Returns the lock file handle (must be closed to release).
    """
    function acquire_registry_lock(models_dir::String; timeout::Int=120)
        lock_path = joinpath(models_dir, ".registry.lock")
        mkpath(models_dir)

        start_time = time()
        while true
            try
                # Try to create lock file exclusively
                lock_file = open(lock_path, "w"; lock=true)
                write(lock_file, "$(getpid())\n$(now())\n$(gethostname())")
                flush(lock_file)
                return lock_file
            catch e
                if time() - start_time > timeout
                    error("Timeout waiting for registry lock after $(timeout)s")
                end
                sleep(0.1 + rand() * 0.3)  # Random backoff to reduce contention
            end
        end
    end

    """
    Release the registry lock.
    """
    function release_registry_lock(lock_file::IO)
        close(lock_file)
    end

    """
    Train and save a model with proper locking.
    """
    function train_and_save_locked(
        models_dir::String,
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

        # Check if model already exists (quick check without lock)
        if model_exists(models_dir; K=K, model_type=model_type,
                       censoring_lower=censoring_lower, censoring_upper=censoring_upper)
            verbose && println("[Worker $worker_id@$hostname] Model already exists: $model_type K=$K censor=($censoring_lower,$censoring_upper)")
            return nothing
        end

        verbose && println("[Worker $worker_id@$hostname] Training $model_type K=$K censor=($censoring_lower,$censoring_upper)...")
        train_start = now()

        # Train the model (this is the time-consuming part, no lock needed)
        if model_type == :nbe
            point_est, ci_est = train_nbe(K;
                width=width,
                n_hidden=n_hidden,
                train_size=train_size,
                m=m,
                epochs=epochs_nbe,
                censoring_lower=censoring_lower,
                censoring_upper=censoring_upper,
                savepath=nothing,  # Don't save yet
                verbose=verbose
            )

            train_elapsed = now() - train_start
            verbose && println("[Worker $worker_id@$hostname] Training complete in $train_elapsed, acquiring lock...")

            # Now acquire lock and save
            lock_file = acquire_registry_lock(models_dir)
            try
                # Double-check model doesn't exist (another worker might have saved it)
                if model_exists(models_dir; K=K, model_type=model_type,
                               censoring_lower=censoring_lower, censoring_upper=censoring_upper)
                    verbose && println("[Worker $worker_id@$hostname] Model was saved by another worker, skipping")
                    return nothing
                end

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

                model_id = save_nbe_model(models_dir, point_est, ci_est, config)
                verbose && println("[Worker $worker_id@$hostname] Saved NBE model ID=$model_id")
                return model_id
            finally
                release_registry_lock(lock_file)
            end

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
                savepath=nothing,  # Don't save yet
                verbose=verbose
            )

            train_elapsed = now() - train_start
            verbose && println("[Worker $worker_id@$hostname] Training complete in $train_elapsed, acquiring lock...")

            # Now acquire lock and save
            lock_file = acquire_registry_lock(models_dir)
            try
                # Double-check model doesn't exist
                if model_exists(models_dir; K=K, model_type=model_type,
                               censoring_lower=censoring_lower, censoring_upper=censoring_upper)
                    verbose && println("[Worker $worker_id@$hostname] Model was saved by another worker, skipping")
                    return nothing
                end

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

                model_id = save_model(models_dir, estimator, config)
                verbose && println("[Worker $worker_id@$hostname] Saved NPE model ID=$model_id")
                return model_id
            finally
                release_registry_lock(lock_file)
            end
        else
            error("Unknown model type: $model_type")
        end
    end
end

#=
Training configuration
=#

"""
Generate list of training jobs as named tuples.
"""
function generate_training_jobs(;
    K_values::Vector{Int}=[3, 4, 5, 6, 7],
    model_types::Vector{Symbol}=[:nbe, :npe],
    censoring_configs::Vector{Tuple{Int,Int}}=[(0, 0)],  # (lower, upper) pairs
    kwargs...
)
    jobs = []
    for K in K_values
        for model_type in model_types
            for (censoring_lower, censoring_upper) in censoring_configs
                push!(jobs, (
                    model_type=model_type,
                    K=K,
                    censoring_lower=censoring_lower,
                    censoring_upper=censoring_upper,
                    kwargs...
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
    mkpath(models_dir)

    jobs = generate_training_jobs(;
        K_values=K_values,
        model_types=model_types,
        censoring_configs=censoring_configs
    )

    println("=" ^ 70)
    println("Distributed Training Configuration")
    println("=" ^ 70)
    println("Output directory: $models_dir")
    println("Workers: $(nworkers())")
    println("Total jobs: $(length(jobs))")
    println("K values: $K_values")
    println("Model types: $model_types")
    println("Censoring configs: $censoring_configs")
    println("Architecture: width=$width, n_hidden=$n_hidden, encoding_dim=$encoding_dim")
    println("Training: train_size=$train_size, epochs_nbe=$epochs_nbe, epochs_npe=$epochs_npe, m=$m")
    if haskey(ENV, "SLURM_JOB_ID")
        println("SLURM Job: $(ENV["SLURM_JOB_ID"])")
        println("SLURM Nodes: $(get(ENV, "SLURM_JOB_NODELIST", "unknown"))")
    end
    println("=" ^ 70)
    println()

    start_time = now()

    # Use pmap for parallel execution with error handling
    results = pmap(jobs; on_error=ex -> ex) do job
        try
            model_id = train_and_save_locked(
                models_dir,
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
            return (job=job, model_id=model_id, error=nothing)
        catch e
            return (job=job, model_id=nothing, error=sprint(showerror, e, catch_backtrace()))
        end
    end

    elapsed = now() - start_time

    # Summary
    println()
    println("=" ^ 70)
    println("Training Complete")
    println("=" ^ 70)
    println("Elapsed time: $elapsed")

    # Handle results (some might be exceptions from on_error)
    processed_results = map(results) do r
        if r isa Exception
            return (job=nothing, model_id=nothing, error=sprint(showerror, r))
        else
            return r
        end
    end

    successes = filter(r -> r.error === nothing && r.model_id !== nothing, processed_results)
    skipped = filter(r -> r.error === nothing && r.model_id === nothing && r.job !== nothing, processed_results)
    failures = filter(r -> r.error !== nothing, processed_results)

    println("Trained: $(length(successes))")
    println("Skipped (already exist): $(length(skipped))")
    println("Failed: $(length(failures))")

    if !isempty(failures)
        println("\nFailures:")
        for r in failures
            job_desc = r.job !== nothing ? "$(r.job.model_type) K=$(r.job.K)" : "unknown job"
            println("  $job_desc:")
            # Print first few lines of error
            error_lines = split(r.error, '\n')
            for line in error_lines[1:min(5, length(error_lines))]
                println("    $line")
            end
            if length(error_lines) > 5
                println("    ... ($(length(error_lines) - 5) more lines)")
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

    return results
end

#=
Main execution
=#

# Default training configuration - modify these as needed
const MODELS_DIR = get(ENV, "NEURALMSE_MODELS_DIR",
                       joinpath(dirname(dirname(@__FILE__)), "trained_models"))

const K_VALUES = [3, 4, 5, 6, 7]
const MODEL_TYPES = [:nbe, :npe]
const CENSORING_CONFIGS = [(0, 0)]  # Add more like [(0,0), (1,5), (1,10)] for censoring

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
