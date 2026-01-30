#=
Artifact management for pretrained NeuralMSE models.

This module provides:
- Lazy downloading of pretrained models from GitHub releases
- Model lookup and listing functions
- Development mode support via NEURALMSE_MODELS_PATH environment variable
=#

using DataFrames
using Pkg.Artifacts: artifact_meta, artifact_path, ensure_artifact_installed
using Pkg.TOML

# Artifact name for pretrained models
const ARTIFACT_NAME = "neuralmse_models_v2"

# Cache for artifacts availability check
const _artifacts_available = Ref{Union{Bool, Nothing}}(nothing)

#=
Artifact path management
=#

"""
    _check_artifacts_available() -> Bool

Check if the artifact system is properly configured (valid Artifacts.toml).
"""
function _check_artifacts_available()
    if _artifacts_available[] === nothing
        try
            artifacts_toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
            if isfile(artifacts_toml)
                content = read(artifacts_toml, String)
                # Check for placeholder hash (all zeros)
                if occursin("0000000000000000000000000000000000000000", content)
                    _artifacts_available[] = false
                else
                    _artifacts_available[] = true
                end
            else
                _artifacts_available[] = false
            end
        catch
            _artifacts_available[] = false
        end
    end
    return _artifacts_available[]
end

"""
    _get_artifact_path() -> String

Get the artifact path using runtime API (not compile-time macro).
"""
function _get_artifact_path()
    artifacts_toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
    meta = artifact_meta(ARTIFACT_NAME, artifacts_toml)
    if meta === nothing
        error("Artifact '$ARTIFACT_NAME' not found in Artifacts.toml")
    end
    hash = Base.SHA1(meta["git-tree-sha1"])
    ensure_artifact_installed(ARTIFACT_NAME, meta, artifacts_toml)
    return artifact_path(hash)
end

"""
    get_models_path() -> String

Get the path to the pretrained models directory.

The path is determined in the following order:
1. If `NEURALMSE_MODELS_PATH` environment variable is set, use that path
2. Otherwise, try to load from the lazy artifact (downloads if needed)

For development/training, set `NEURALMSE_MODELS_PATH` to your local models directory.

# Returns
The path to the models directory containing `trained_models.jld2` and model files.

# Example
```julia
# Use pretrained models (downloads automatically)
models_path = get_models_path()

# Or set environment variable for local development:
# ENV["NEURALMSE_MODELS_PATH"] = "/path/to/trained_models"
# models_path = get_models_path()
```
"""
function get_models_path()
    # Check for development/custom path first
    custom_path = get(ENV, "NEURALMSE_MODELS_PATH", "")
    if !isempty(custom_path)
        if !isdir(custom_path)
            error("NEURALMSE_MODELS_PATH is set to '$custom_path' but directory does not exist")
        end
        return custom_path
    end

    # Check if artifacts are available
    if !_check_artifacts_available()
        error("""
            Pretrained models not available.

            The Artifacts.toml has placeholder values (models not yet uploaded to GitHub).

            For development/training, set the models path environment variable:
                ENV["NEURALMSE_MODELS_PATH"] = "/path/to/trained_models"

            Or run Julia with:
                NEURALMSE_MODELS_PATH=/path/to/trained_models julia ...
            """)
    end

    # Try to use the artifact (runtime resolution)
    try
        return _get_artifact_path()
    catch e
        error("""
            Failed to download pretrained models.

            Check your internet connection and try again.

            For development/training, set the models path:
                ENV["NEURALMSE_MODELS_PATH"] = "/path/to/trained_models"

            Original error: $e
            """)
    end
end

"""
    ensure_models_available() -> String

Ensure the pretrained models are downloaded and return the path.

This is an alias for `get_models_path()` with a more descriptive name.

# Returns
The path to the models directory.
"""
ensure_models_available() = get_models_path()

#=
Model listing and discovery
=#

"""
    list_available_models() -> DataFrame

List all available pretrained models with their configurations.

Downloads the models artifact if not already present.

# Returns
A DataFrame with columns: id, model_type, K, width, n_hidden, encoding_dim,
censoring_lower, censoring_upper, train_size, trained_at

# Example
```julia
models = list_available_models()

# Filter for K=5 NPE models
filter(row -> row.K == 5 && row.model_type == :npe, models)

# Find all models without censoring
filter(row -> row.censoring_upper == 0, models)
```
"""
function list_available_models()
    models_path = get_models_path()
    return list_models(models_path)
end

"""
    find_pretrained_model(; K::Int, model_type::Symbol,
                          censoring_lower::Int=0, censoring_upper::Int=0) -> Union{Int, Nothing}

Find a pretrained model ID by configuration.

# Arguments
- `K`: Number of lists
- `model_type`: Type of estimator (`:nbe_point`, `:nbe_interval`, or `:npe`)
- `censoring_lower`: Lower censoring bound (default: 0)
- `censoring_upper`: Upper censoring bound (default: 0)

# Returns
The model ID if found, or `nothing` if no matching model exists.

# Example
```julia
model_id = find_pretrained_model(K=5, model_type=:nbe_point)
if model_id !== nothing
    println("Found model with ID: \$model_id")
end
```
"""
function find_pretrained_model(; K::Int, model_type::Symbol,
                               censoring_lower::Int=0, censoring_upper::Int=0)
    models_path = get_models_path()
    return find_model(models_path; K=K, model_type=model_type,
                      censoring_lower=censoring_lower,
                      censoring_upper=censoring_upper)
end

"""
    load_pretrained_model(; K::Int, model_type::Symbol,
                          censoring_lower::Int=0, censoring_upper::Int=0) -> Tuple{Any, ModelConfig}

Load a pretrained model by configuration.

# Arguments
- `K`: Number of lists
- `model_type`: Type of estimator (`:nbe_point`, `:nbe_interval`, or `:npe`)
- `censoring_lower`: Lower censoring bound (default: 0)
- `censoring_upper`: Upper censoring bound (default: 0)

# Returns
A tuple (estimator, config) for the matching model.

# Example
```julia
estimator, config = load_pretrained_model(K=5, model_type=:nbe_point)
```
"""
function load_pretrained_model(; K::Int, model_type::Symbol,
                               censoring_lower::Int=0, censoring_upper::Int=0)
    models_path = get_models_path()
    return load_model(models_path; K=K, model_type=model_type,
                      censoring_lower=censoring_lower,
                      censoring_upper=censoring_upper)
end

"""
    load_pretrained_model(model_id::Int) -> Tuple{Any, ModelConfig}

Load a pretrained model by its ID.

# Arguments
- `model_id`: The model's unique ID

# Returns
A tuple (estimator, config) for the model.

# Example
```julia
estimator, config = load_pretrained_model(42)
```
"""
function load_pretrained_model(model_id::Int)
    models_path = get_models_path()
    return load_model(models_path, model_id)
end

"""
    pretrained_model_exists(; K::Int, model_type::Symbol,
                            censoring_lower::Int=0, censoring_upper::Int=0) -> Bool

Check if a pretrained model with the given configuration exists.

# Example
```julia
if pretrained_model_exists(K=5, model_type=:npe)
    println("NPE model for K=5 is available")
end
```
"""
function pretrained_model_exists(; K::Int, model_type::Symbol,
                                 censoring_lower::Int=0, censoring_upper::Int=0)
    models_path = get_models_path()
    return model_exists(models_path; K=K, model_type=model_type,
                        censoring_lower=censoring_lower,
                        censoring_upper=censoring_upper)
end

#=
Utility functions
=#

"""
    print_available_models()

Print a summary of all available pretrained models.

# Example
```julia
print_available_models()
# Output:
# Available NeuralMSE Pretrained Models
# =====================================
# NBE Point Estimators: 45 models (K: 3-10)
# NBE Interval Estimators: 45 models (K: 3-10)
# NPE Estimators: 30 models (K: 3-6)
```
"""
function print_available_models()
    df = list_available_models()

    println("Available NeuralMSE Pretrained Models")
    println("=====================================")

    for mtype in [:nbe_point, :nbe_interval, :npe]
        subset = filter(row -> row.model_type == mtype, df)
        if nrow(subset) > 0
            K_range = "$(minimum(subset.K))-$(maximum(subset.K))"
            type_name = mtype == :nbe_point ? "NBE Point Estimators" :
                        mtype == :nbe_interval ? "NBE Interval Estimators" :
                        "NPE Estimators"
            println("$type_name: $(nrow(subset)) models (K: $K_range)")
        end
    end
end
