#=
Collect Temporary Models Script

This script collects trained models from the temp directory (.temp_training)
that were saved during a distributed training run that didn't complete.

Usage:
    julia --project scripts/collect_temp_models.jl [--cleanup]

Options:
    --cleanup    Remove temp files after successful collection
=#

using NeuralMSE
using NeuralMSE: ModelConfig, save_model, save_nbe_model, model_exists, list_models
using JLD2
using DataFrames

const MODELS_DIR = get(ENV, "NEURALMSE_MODELS_PATH",
                       joinpath(dirname(dirname(@__FILE__)), "models"))
const TEMP_DIR = joinpath(MODELS_DIR, ".temp_training")
const CLEANUP = "--cleanup" in ARGS

function collect_temp_models(temp_dir::String, models_dir::String; cleanup::Bool=false)
    println("=" ^ 70)
    println("Collecting Temporary Models")
    println("=" ^ 70)
    println("Temp directory: $temp_dir")
    println("Models directory: $models_dir")
    println("Cleanup after collection: $cleanup")
    println("=" ^ 70)
    println()

    if !isdir(temp_dir)
        println("ERROR: Temp directory does not exist: $temp_dir")
        println("Nothing to collect.")
        return 0
    end

    # Find all job files
    job_files = filter(f -> startswith(f, "job_") && endswith(f, ".jld2"), readdir(temp_dir))

    if isempty(job_files)
        println("No temp model files found in $temp_dir")
        return 0
    end

    println("Found $(length(job_files)) temp model files")
    println()

    mkpath(models_dir)

    collected = 0
    skipped_exists = 0
    skipped_error = 0
    collected_files = String[]

    for job_file in sort(job_files)
        filepath = joinpath(temp_dir, job_file)
        print("Processing $job_file... ")

        try
            jldopen(filepath, "r") do f
                model_type = f["model_type"]
                config = f["config"]

                # Check if model already exists in registry
                if model_exists(models_dir; K=config.K, model_type=config.model_type,
                               censoring_lower=config.censoring_lower,
                               censoring_upper=config.censoring_upper)
                    println("SKIPPED (already in registry)")
                    skipped_exists += 1
                    push!(collected_files, filepath)  # Mark for cleanup even if skipped
                    return
                end

                if model_type == :nbe
                    point_est = f["point_estimator"]
                    interval_est = f["interval_estimator"]
                    model_id = save_nbe_model(models_dir, point_est, interval_est, config)
                    println("COLLECTED NBE model ID=$model_id (K=$(config.K), censor=$(config.censoring_lower),$(config.censoring_upper))")
                else
                    estimator = f["estimator"]
                    model_id = save_model(models_dir, estimator, config)
                    println("COLLECTED NPE model ID=$model_id (K=$(config.K), censor=$(config.censoring_lower),$(config.censoring_upper))")
                end

                collected += 1
                push!(collected_files, filepath)
            end
        catch e
            println("ERROR: $(sprint(showerror, e))")
            skipped_error += 1
        end
    end

    println()
    println("=" ^ 70)
    println("Collection Summary")
    println("=" ^ 70)
    println("  Collected: $collected")
    println("  Skipped (already exists): $skipped_exists")
    println("  Skipped (errors): $skipped_error")
    println("  Total processed: $(collected + skipped_exists + skipped_error)")

    # Show registry status
    println()
    println("Models in registry:")
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

    # Cleanup if requested
    if cleanup && !isempty(collected_files)
        println()
        println("Cleaning up $(length(collected_files)) processed temp files...")
        for f in collected_files
            try
                rm(f)
            catch e
                println("  Warning: Could not remove $f: $e")
            end
        end

        # Remove temp dir if empty
        remaining = readdir(temp_dir)
        if isempty(remaining)
            rm(temp_dir)
            println("Removed empty temp directory")
        else
            println("$(length(remaining)) files remaining in temp directory (errors or new files)")
        end
    elseif !cleanup && collected > 0
        println()
        println("Tip: Run with --cleanup to remove processed temp files")
    end

    println("=" ^ 70)

    return collected
end

# Run collection
collect_temp_models(TEMP_DIR, MODELS_DIR; cleanup=CLEANUP)
