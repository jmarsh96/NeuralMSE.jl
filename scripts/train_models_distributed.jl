#=
Distributed Training Script for NeuralMSE Models

This script trains multiple NBE and NPE models in parallel using Julia's
distributed computing capabilities. It uses file locking to prevent race
conditions when multiple workers try to update the model registry simultaneously.

Usage:
    julia -p 4 scripts/train_models_distributed.jl          # 4 worker processes
    julia -p auto scripts/train_models_distributed.jl       # Use all available cores

Or from Julia REPL:
    using Distributed
    addprocs(4)
    include("scripts/train_models_distributed.jl")
=#

using Distributed

# Add workers if not already added (when running as script)
if nworkers() == 1
    # Default to number of physical cores - 1, minimum 1
    n_workers = max(1, Sys.CPU_THREADS ÷ 2 - 1)
    println("Adding $n_workers worker processes...")
    addprocs(n_workers)
end

println("Running with $(nworkers()) workers")

# Load packages on all workers
@everywhere begin
    using Pkg
    Pkg.activate(dirname(dirname(@__FILE__)))

    using NeuralMSE
    using Dates
    using FileWatching  # For file locking
end

#=
File-based locking for safe concurrent registry updates
=#

@everywhere begin
    """
    Acquire an exclusive lock on the model registry.
    Returns the lock file handle (must be closed to release).
    """
    function acquire_registry_lock(models_dir::String; timeout::Int=60)
        lock_path = joinpath(models_dir, ".registry.lock")
        mkpath(models_dir)

        start_time = time()
        while true
            try
                # Try to create lock file exclusively
                lock_file = open(lock_path, "w"; lock=true)
                write(lock_file, "$(getpid())\n$(now())")
                return lock_file
            catch e
                if time() - start_time > timeout
                    error("Timeout waiting for registry lock after $(timeout)s")
                end
                sleep(0.1 + rand() * 0.2)  # Random backoff to reduce contention
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

        # Check if model already exists (quick check without lock)
        if model_exists(models_dir; K=K, model_type=model_type,
                       censoring_lower=censoring_lower, censoring_upper=censoring_upper)
            verbose && println("[Worker $worker_id] Model already exists: $model_type K=$K censor=($censoring_lower,$censoring_upper)")
            return nothing
        end

        verbose && println("[Worker $worker_id] Training $model_type K=$K censor=($censoring_lower,$censoring_upper)...")

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

            # Now acquire lock and save
            lock_file = acquire_registry_lock(models_dir)
            try
                # Double-check model doesn't exist (another worker might have saved it)
                if model_exists(models_dir; K=K, model_type=model_type,
                               censoring_lower=censoring_lower, censoring_upper=censoring_upper)
                    verbose && println("[Worker $worker_id] Model was saved by another worker, skipping")
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
                verbose && println("[Worker $worker_id] Saved NBE model ID=$model_id")
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

            # Now acquire lock and save
            lock_file = acquire_registry_lock(models_dir)
            try
                # Double-check model doesn't exist
                if model_exists(models_dir; K=K, model_type=model_type,
                               censoring_lower=censoring_lower, censoring_upper=censoring_upper)
                    verbose && println("[Worker $worker_id] Model was saved by another worker, skipping")
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
                verbose && println("[Worker $worker_id] Saved NPE model ID=$model_id")
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

    println("=" ^ 60)
    println("Distributed Training Configuration")
    println("=" ^ 60)
    println("Output directory: $models_dir")
    println("Workers: $(nworkers())")
    println("Total jobs: $(length(jobs))")
    println("K values: $K_values")
    println("Model types: $model_types")
    println("Censoring configs: $censoring_configs")
    println("Architecture: width=$width, n_hidden=$n_hidden, encoding_dim=$encoding_dim")
    println("Training: train_size=$train_size, epochs_nbe=$epochs_nbe, epochs_npe=$epochs_npe")
    println("=" ^ 60)
    println()

    start_time = now()

    # Use pmap for parallel execution
    results = pmap(jobs) do job
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
            return (job=job, model_id=nothing, error=sprint(showerror, e))
        end
    end

    elapsed = now() - start_time

    # Summary
    println()
    println("=" ^ 60)
    println("Training Complete")
    println("=" ^ 60)
    println("Elapsed time: $elapsed")

    successes = filter(r -> r.error === nothing && r.model_id !== nothing, results)
    skipped = filter(r -> r.error === nothing && r.model_id === nothing, results)
    failures = filter(r -> r.error !== nothing, results)

    println("Trained: $(length(successes))")
    println("Skipped (already exist): $(length(skipped))")
    println("Failed: $(length(failures))")

    if !isempty(failures)
        println("\nFailures:")
        for r in failures
            println("  $(r.job.model_type) K=$(r.job.K): $(r.error)")
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
    catch
        println("  (Could not read registry)")
    end

    println("=" ^ 60)

    return results
end

#=
Main execution
=#

if abspath(PROGRAM_FILE) == @__FILE__
    # Default training configuration
    models_dir = joinpath(dirname(dirname(@__FILE__)), "trained_models")

    # Example: Train models for K=3,4,5 with no censoring
    # Adjust these parameters as needed
    train_all_models(
        models_dir;
        K_values=[3, 4, 5],
        model_types=[:nbe, :npe],
        censoring_configs=[(0, 0)],  # Add more like [(0,0), (1,5), (1,10)] for censoring
        width=256,
        n_hidden=3,
        encoding_dim=128,
        train_size=10000,
        epochs_nbe=100,
        epochs_npe=200,
        verbose=true
    )
end
