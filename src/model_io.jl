#=
Model I/O utilities for NeuralMSE using JLD2 format.

This module provides:
- ModelConfig struct for storing model metadata and priors
- Centralized model registry (trained_models.jld2)
- Save/load functions for trained estimators
=#

using JLD2
using Dates
using DataFrames

#=
Configuration struct
=#

"""
    ModelConfig

Configuration metadata for a trained NeuralMSE model.

# Fields
- `model_type::Symbol`: Type of estimator (`:nbe` or `:npe`)
  - `:nbe` models contain both a PointEstimator and IntervalEstimator
  - `:npe` models contain a PosteriorEstimator
- `K::Int`: Number of lists
- `width::Int`: Width of MLP hidden layers
- `n_hidden::Int`: Number of hidden layers
- `encoding_dim::Union{Nothing,Int}`: Encoding dimension for NPE (nothing for NBE)
- `train_size::Int`: Number of training samples used
- `m::Int`: Number of replicates per parameter sample during training
- `censoring_lower::Int`: Lower censoring bound
- `censoring_upper::Int`: Upper censoring bound
- `intercept_prior::NamedTuple`: Prior for intercept, e.g., `(type=:Uniform, a=1.0, b=10.0)`
- `beta_prior::NamedTuple`: Prior for betas, e.g., `(type=:Normal, μ=0.0, σ=4.0)`
- `gamma_prior::NamedTuple`: Prior for gammas, e.g., `(type=:Normal, μ=0.0, σ=4.0)`
- `trained_at::DateTime`: Timestamp when model was trained

# Example
```julia
config = ModelConfig(
    model_type=:nbe,
    K=5,
    width=256,
    n_hidden=3,
    intercept_prior=(type=:Uniform, a=1.0, b=10.0),
    beta_prior=(type=:Normal, μ=0.0, σ=4.0),
    gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
)
```
"""
Base.@kwdef struct ModelConfig
    model_type::Symbol
    K::Int
    width::Int
    n_hidden::Int
    encoding_dim::Union{Nothing,Int} = nothing
    train_size::Int = 10000
    m::Int = 1
    censoring_lower::Int = 0
    censoring_upper::Int = 0
    intercept_prior::NamedTuple = (type=:Uniform, a=1.0, b=10.0)
    beta_prior::NamedTuple = (type=:Normal, μ=0.0, σ=4.0)
    gamma_prior::NamedTuple = (type=:Normal, μ=0.0, σ=4.0)
    trained_at::DateTime = now()
end

# Define equality for ModelConfig
function Base.:(==)(a::ModelConfig, b::ModelConfig)
    return a.model_type == b.model_type &&
           a.K == b.K &&
           a.width == b.width &&
           a.n_hidden == b.n_hidden &&
           a.encoding_dim == b.encoding_dim &&
           a.train_size == b.train_size &&
           a.m == b.m &&
           a.censoring_lower == b.censoring_lower &&
           a.censoring_upper == b.censoring_upper &&
           a.intercept_prior == b.intercept_prior &&
           a.beta_prior == b.beta_prior &&
           a.gamma_prior == b.gamma_prior
end

#=
Registry file management
=#

const REGISTRY_FILENAME = "trained_models.jld2"
const REGISTRY_VERSION = 1

"""
    _get_registry_path(models_dir::String) -> String

Get the path to the model registry file.
"""
_get_registry_path(models_dir::String) = joinpath(models_dir, REGISTRY_FILENAME)

"""
    _get_model_path(models_dir::String, model_id::Int) -> String

Get the path to a specific model file.
"""
_get_model_path(models_dir::String, model_id::Int) = joinpath(models_dir, "model_$model_id.jld2")

"""
    load_model_registry(models_dir::String) -> Dict{Int, ModelConfig}

Load the model registry from the models directory.

Returns an empty Dict if the registry doesn't exist.
"""
function load_model_registry(models_dir::String)
    registry_path = _get_registry_path(models_dir)
    if !isfile(registry_path)
        return Dict{Int, ModelConfig}()
    end

    data = JLD2.load(registry_path)
    return data["models"]
end

"""
    _load_registry_data(models_dir::String) -> Dict

Load the full registry data including version and next_id.
"""
function _load_registry_data(models_dir::String)
    registry_path = _get_registry_path(models_dir)
    if !isfile(registry_path)
        return Dict(
            "version" => REGISTRY_VERSION,
            "models" => Dict{Int, ModelConfig}(),
            "next_id" => 1
        )
    end
    return JLD2.load(registry_path)
end

"""
    _save_registry_data(models_dir::String, data::Dict)

Save the registry data to disk.
"""
function _save_registry_data(models_dir::String, data::Dict)
    registry_path = _get_registry_path(models_dir)
    JLD2.save(registry_path, data)
end

#=
Model save/load functions
=#

"""
    save_model(models_dir::String, estimator, config::ModelConfig) -> Int

Save a trained estimator with its configuration to the models directory.

The model is assigned a unique ID and added to the registry.

# Arguments
- `models_dir`: Directory containing the model registry and model files
- `estimator`: The trained PosteriorEstimator (for NPE models)
- `config`: ModelConfig with training metadata and priors

# Returns
The assigned model ID (Int).

# Example
```julia
model_id = save_model("models/", npe_estimator, config)
println("Saved model with ID: \$model_id")
```
"""
function save_model(models_dir::String, estimator, config::ModelConfig)
    # Ensure directory exists
    mkpath(models_dir)

    # Load or initialize registry
    registry_data = _load_registry_data(models_dir)

    # Get next ID and increment
    model_id = registry_data["next_id"]
    registry_data["next_id"] = model_id + 1

    # Add config to registry
    registry_data["models"][model_id] = config

    # Save the estimator
    model_path = _get_model_path(models_dir, model_id)
    JLD2.save(model_path, Dict("estimator" => estimator))

    # Save updated registry
    _save_registry_data(models_dir, registry_data)

    return model_id
end

"""
    save_nbe_model(models_dir::String, point_estimator, interval_estimator, config::ModelConfig) -> Int

Save an NBE model containing both point and interval estimators.

NBE models are saved as a single unit with both estimators in the same file.

# Arguments
- `models_dir`: Directory containing the model registry and model files
- `point_estimator`: The trained PointEstimator
- `interval_estimator`: The trained IntervalEstimator
- `config`: ModelConfig with model_type=:nbe

# Returns
The assigned model ID (Int).

# Example
```julia
model_id = save_nbe_model("models/", point_est, ci_est, config)
```
"""
function save_nbe_model(models_dir::String, point_estimator, interval_estimator, config::ModelConfig)
    if config.model_type != :nbe
        error("save_nbe_model requires config.model_type == :nbe, got $(config.model_type)")
    end

    # Ensure directory exists
    mkpath(models_dir)

    # Load or initialize registry
    registry_data = _load_registry_data(models_dir)

    # Get next ID and increment
    model_id = registry_data["next_id"]
    registry_data["next_id"] = model_id + 1

    # Add config to registry
    registry_data["models"][model_id] = config

    # Save both estimators in the same file
    model_path = _get_model_path(models_dir, model_id)
    JLD2.save(model_path, Dict(
        "point_estimator" => point_estimator,
        "interval_estimator" => interval_estimator
    ))

    # Save updated registry
    _save_registry_data(models_dir, registry_data)

    return model_id
end

"""
    load_model(models_dir::String, model_id::Int) -> Tuple{Any, ModelConfig}

Load a model by its ID.

# Arguments
- `models_dir`: Directory containing the model registry and model files
- `model_id`: The model's unique ID

# Returns
For NBE models (model_type=:nbe):
- A tuple ((point_estimator, interval_estimator), config)

For NPE models (model_type=:npe):
- A tuple (estimator, config)

# Example
```julia
# Load NBE model
(point_est, ci_est), config = load_model("models/", 42)

# Load NPE model
estimator, config = load_model("models/", 43)
```
"""
function load_model(models_dir::String, model_id::Int)
    # Load registry to get config
    registry = load_model_registry(models_dir)

    if !haskey(registry, model_id)
        error("Model ID $model_id not found in registry at $models_dir")
    end

    config = registry[model_id]

    # Load estimator(s)
    model_path = _get_model_path(models_dir, model_id)
    if !isfile(model_path)
        error("Model file not found: $model_path")
    end

    data = JLD2.load(model_path)

    # Handle NBE models (contain both point and interval estimators)
    if config.model_type == :nbe
        point_estimator = data["point_estimator"]
        interval_estimator = data["interval_estimator"]
        return (point_estimator, interval_estimator), config
    else
        # NPE or other single-estimator models
        estimator = data["estimator"]
        return estimator, config
    end
end

"""
    load_model(models_dir::String; K::Int, model_type::Symbol,
               censoring_lower::Int=0, censoring_upper::Int=0) -> Tuple{Any, ModelConfig}

Load a model by matching configuration parameters.

Finds the first model in the registry that matches all specified parameters.

# Arguments
- `models_dir`: Directory containing the model registry and model files
- `K`: Number of lists
- `model_type`: Type of estimator (`:nbe` or `:npe`)
- `censoring_lower`: Lower censoring bound (default: 0)
- `censoring_upper`: Upper censoring bound (default: 0)

# Returns
For NBE models: A tuple ((point_estimator, interval_estimator), config)
For NPE models: A tuple (estimator, config)

# Example
```julia
(point_est, ci_est), config = load_model("models/"; K=5, model_type=:nbe)
npe_estimator, config = load_model("models/"; K=5, model_type=:npe)
```
"""
function load_model(models_dir::String;
                    K::Int,
                    model_type::Symbol,
                    censoring_lower::Int=0,
                    censoring_upper::Int=0)
    registry = load_model_registry(models_dir)

    for (model_id, config) in registry
        if config.K == K &&
           config.model_type == model_type &&
           config.censoring_lower == censoring_lower &&
           config.censoring_upper == censoring_upper
            return load_model(models_dir, model_id)
        end
    end

    error("No model found matching: K=$K, model_type=$model_type, " *
          "censoring_lower=$censoring_lower, censoring_upper=$censoring_upper")
end

"""
    find_model(models_dir::String; K::Int, model_type::Symbol,
               censoring_lower::Int=0, censoring_upper::Int=0) -> Union{Int, Nothing}

Find a model ID by matching configuration parameters.

# Returns
The model ID if found, or `nothing` if no match exists.

# Example
```julia
model_id = find_model("models/"; K=5, model_type=:npe)
if model_id !== nothing
    estimator, config = load_model("models/", model_id)
end
```
"""
function find_model(models_dir::String;
                    K::Int,
                    model_type::Symbol,
                    censoring_lower::Int=0,
                    censoring_upper::Int=0)
    registry = load_model_registry(models_dir)

    for (model_id, config) in registry
        if config.K == K &&
           config.model_type == model_type &&
           config.censoring_lower == censoring_lower &&
           config.censoring_upper == censoring_upper
            return model_id
        end
    end

    return nothing
end

"""
    list_models(models_dir::String) -> DataFrame

List all available models with their configurations.

# Returns
A DataFrame with columns: id, model_type, K, width, n_hidden, encoding_dim,
censoring_lower, censoring_upper, train_size, trained_at

# Example
```julia
models = list_models("models/")
filter(row -> row.K == 5 && row.model_type == :npe, models)
```
"""
function list_models(models_dir::String)
    registry = load_model_registry(models_dir)

    if isempty(registry)
        return DataFrame(
            id=Int[],
            model_type=Symbol[],
            K=Int[],
            width=Int[],
            n_hidden=Int[],
            encoding_dim=Union{Nothing,Int}[],
            censoring_lower=Int[],
            censoring_upper=Int[],
            train_size=Int[],
            trained_at=DateTime[]
        )
    end

    rows = []
    for (model_id, config) in registry
        push!(rows, (
            id=model_id,
            model_type=config.model_type,
            K=config.K,
            width=config.width,
            n_hidden=config.n_hidden,
            encoding_dim=config.encoding_dim,
            censoring_lower=config.censoring_lower,
            censoring_upper=config.censoring_upper,
            train_size=config.train_size,
            trained_at=config.trained_at
        ))
    end

    return DataFrame(rows)
end

"""
    remove_model(models_dir::String, model_id::Int)

Remove a model from the registry and delete its file.

# Arguments
- `models_dir`: Directory containing the model registry and model files
- `model_id`: The model's unique ID to remove

# Example
```julia
remove_model("models/", 42)
```
"""
function remove_model(models_dir::String, model_id::Int)
    # Load registry
    registry_data = _load_registry_data(models_dir)

    if !haskey(registry_data["models"], model_id)
        error("Model ID $model_id not found in registry")
    end

    # Remove from registry
    delete!(registry_data["models"], model_id)

    # Delete model file
    model_path = _get_model_path(models_dir, model_id)
    if isfile(model_path)
        rm(model_path)
    end

    # Save updated registry
    _save_registry_data(models_dir, registry_data)
end

"""
    model_exists(models_dir::String; K::Int, model_type::Symbol,
                 censoring_lower::Int=0, censoring_upper::Int=0) -> Bool

Check if a model with the given configuration exists.

# Example
```julia
if !model_exists("models/"; K=5, model_type=:nbe)
    # Train the model
end
```
"""
function model_exists(models_dir::String;
                      K::Int,
                      model_type::Symbol,
                      censoring_lower::Int=0,
                      censoring_upper::Int=0)
    return find_model(models_dir; K=K, model_type=model_type,
                      censoring_lower=censoring_lower,
                      censoring_upper=censoring_upper) !== nothing
end
