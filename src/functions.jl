#=
Internal BSON loading utilities for handling parametric types

BSON cannot deserialize Julia's UnionAll (parametric) types directly because it needs
a concrete DataType to allocate memory. We work around this by:
1. Parsing the raw BSON data
2. Recursively reconstructing nested parametric types by calling their constructors
3. Letting BSON handle concrete types normally
=#

# Set of type names that are parametric and need manual reconstruction
const PARAMETRIC_TYPES = Set([
    ["NeuralEstimators", "PosteriorEstimator"],
    ["NeuralEstimators", "NormalisingFlow"],
    ["NeuralEstimators", "CouplingLayer"],
    ["NeuralEstimators", "AffineCouplingBlock"],
    ["NeuralEstimators", "MLP"],
    ["NeuralEstimators", "ActNorm"],
    ["NeuralEstimators", "Permutation"],
])

"""
    _resolve_type(type_info)

Resolve a BSON type reference to an actual Julia type.
"""
function _resolve_type(type_info)
    names = type_info[:name]
    # Try to resolve from Main first, then from NeuralEstimators module
    for init_module in (Main, NeuralEstimators)
        T = try
            reduce(getfield, Symbol.(names); init=init_module)
        catch
            nothing
        end
        if T !== nothing
            return T
        end
    end
    # If names start with "NeuralEstimators", try direct resolution
    if length(names) >= 2 && names[1] == "NeuralEstimators"
        return try
            getfield(NeuralEstimators, Symbol(names[end]))
        catch
            nothing
        end
    end
    return nothing
end

"""
    _is_parametric_wrapper(type_info)

Check if a type is one of our known parametric wrapper types that need manual reconstruction.
"""
function _is_parametric_wrapper(type_info)
    type_info[:name] in PARAMETRIC_TYPES
end

"""
    _reconstruct_bson(raw, cache=IdDict{Any,Any}())

Recursively reconstruct BSON data, handling parametric types that BSON can't deserialize.
"""
function _reconstruct_bson(raw, cache=IdDict{Any,Any}())
    # Handle non-dict data
    if !(raw isa Dict)
        if raw isa Vector
            return [_reconstruct_bson(x, cache) for x in raw]
        end
        return raw
    end

    # Non-tagged dicts pass through
    if !haskey(raw, :tag)
        return raw
    end

    tag = raw[:tag]

    if tag == "struct"
        type_info = raw[:type]
        data = raw[:data]

        # Check if this is a parametric type we need to handle manually
        if _is_parametric_wrapper(type_info)
            # Recursively reconstruct all data fields
            reconstructed_data = [_reconstruct_bson(d, cache) for d in data]
            # Get the type and call its constructor
            T = _resolve_type(type_info)
            if T !== nothing
                return T(reconstructed_data...)
            end
        end

        # For non-parametric types, let BSON handle it
        return BSON.raise_recursive(raw, cache, NeuralEstimators)

    elseif tag == "array"
        # Arrays use BSON's standard handling
        return BSON.raise_recursive(raw, cache, NeuralEstimators)
    else
        # Other tags use BSON's standard handling
        return BSON.raise_recursive(raw, cache, NeuralEstimators)
    end
end

"""
    _load_point_estimator(path, key)

Load a PointEstimator from a BSON file, working around BSON's parametric type limitations.
"""
function _load_point_estimator(path::String, key::Symbol)
    raw = BSON.parse(path)
    est_raw = raw[key]
    network_raw = est_raw[:data][1]
    network = BSON.raise_recursive(network_raw, IdDict{Any,Any}(), NeuralEstimators)
    return NeuralEstimators.PointEstimator(network)
end

"""
    _load_interval_estimator(path, key)

Load an IntervalEstimator from a BSON file, working around BSON's parametric type limitations.
"""
function _load_interval_estimator(path::String, key::Symbol)
    raw = BSON.parse(path)
    est_raw = raw[key]
    data = est_raw[:data]
    u = BSON.raise_recursive(data[1], IdDict{Any,Any}(), NeuralEstimators)
    v = BSON.raise_recursive(data[2], IdDict{Any,Any}(), NeuralEstimators)
    c = BSON.raise_recursive(data[3], IdDict{Any,Any}(), NeuralEstimators)
    probs = BSON.raise_recursive(data[4], IdDict{Any,Any}(), NeuralEstimators)
    g = BSON.raise_recursive(data[5], IdDict{Any,Any}(), NeuralEstimators)
    return NeuralEstimators.IntervalEstimator(u, v, c, probs, g)
end

"""
    _load_posterior_estimator(path, key)

Load a PosteriorEstimator from a BSON file, working around BSON's parametric type limitations.
"""
function _load_posterior_estimator(path::String, key::Symbol)
    raw = BSON.parse(path)
    est_raw = raw[key]
    return _reconstruct_bson(est_raw)
end

#=
Model loading functions
=#

function _get_models_path(subdir::String)
    return joinpath(dirname(pathof(NeuralMSE)), "..", "models", subdir)
end

function _load_nbe_estimator(K::Int, censoring_lower::Int, censoring_upper::Int)
    width = 256
    n_hidden = 3
    train_size = 10000
    m = 1
    models_path = _get_models_path("nbe")
    mdl_str = "model_$(K)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    path = joinpath(models_path, mdl_str)
    if !isfile(path)
        error("Model file not found: $mdl_str. Available models may have different parameters.")
    end
    return _load_point_estimator(path, :estimator)
end

function _load_nbe_ci_estimator(K::Int, censoring_lower::Int, censoring_upper::Int)
    width = 256
    n_hidden = 3
    train_size = 10000
    m = 1
    models_path = _get_models_path("nbe")
    mdl_str = "model_ci_$(K)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    path = joinpath(models_path, mdl_str)
    if !isfile(path)
        error("CI model file not found: $mdl_str. Available models may have different parameters.")
    end
    return _load_interval_estimator(path, :ci_estimator)
end

function _load_npe_estimator(K::Int, censoring_lower::Int, censoring_upper::Int; encoding_dim::Int=128)
    width = 256
    n_hidden = 3
    train_size = 10000
    m = 1
    models_path = _get_models_path("npe")
    mdl_str = "model_$(K)_$(width)_$(n_hidden)_$(encoding_dim)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    path = joinpath(models_path, mdl_str)
    if !isfile(path)
        error("NPE model file not found: $mdl_str. Available models may have different parameters.")
    end
    return _load_posterior_estimator(path, :estimator)
end

#=
Data preparation
=#

"""
    prepare_data(data::AbstractVector{<:Real}; censoring::Bool=false)

Prepare input data for neural estimation. Applies log transformation to positive counts
and handles censored observations (indicated by negative values).

# Arguments
- `data`: Vector of counts. Negative values indicate censored observations.
- `censoring`: If true, appends a censoring indicator vector. Required when using models trained with censoring.

# Returns
- A Float32 matrix suitable for input to the neural estimators.
"""
function prepare_data(data::AbstractVector{<:Real}; censoring::Bool=false)
    # Log transform positive counts
    out = [x > 0 ? log(x + 1) : Float32(x) for x in data]

    if censoring
        # Append censoring indicator vector (1 if censored, 0 otherwise)
        append!(out, Float32.(1 * (out .< 0)))
    end

    return reshape(Float32.(out), (length(out), 1))
end

#=
Public API
=#

"""
    infer_nbe(K::Int, censoring_lower::Int, censoring_upper::Int, data::AbstractVector{<:Real})

Perform neural Bayes estimation for multiple systems estimation (MSE) parameters.

# Arguments
- `K`: Number of lists
- `censoring_lower`: Lower censoring bound
- `censoring_upper`: Upper censoring bound (use 0 for no censoring)
- `data`: Vector of observed counts (length should be 2^K - 1). Negative values indicate censored observations.

# Returns
A named tuple with:
- `median`: Posterior median estimates for all parameters (intercept, betas, gammas)
- `ci_lower`: 2.5% quantile (lower bound of 95% CI)
- `ci_upper`: 97.5% quantile (upper bound of 95% CI)

# Example
```julia
K = 5
data = [10, 5, 3, 8, 2, ...]  # 2^5 - 1 = 31 observations
result = infer_nbe(K, 0, 0, data)
result.median  # point estimates
result.ci_lower, result.ci_upper  # 95% credible interval
```
"""
function infer_nbe(K::Int, censoring_lower::Int, censoring_upper::Int, data::AbstractVector{<:Real})
    # Validate input size
    expected_size = 2^K - 1
    if length(data) != expected_size
        error("Expected $expected_size observations for K=$K lists, got $(length(data))")
    end

    # Prepare data (include censoring indicator if model expects it)
    X = prepare_data(data; censoring=(censoring_upper > 0))

    # Load estimators
    point_estimator = _load_nbe_estimator(K, censoring_lower, censoring_upper)
    ci_estimator = _load_nbe_ci_estimator(K, censoring_lower, censoring_upper)

    # Run inference
    median_estimates = vec(point_estimator(X))
    ci_estimates = ci_estimator(X)

    # CI estimator returns stacked lower and upper bounds
    n_params = length(median_estimates)
    ci_lower = vec(ci_estimates[1:n_params, :])
    ci_upper = vec(ci_estimates[n_params+1:end, :])

    return (median=median_estimates, ci_lower=ci_lower, ci_upper=ci_upper)
end

"""
    infer_npe(K::Int, censoring_lower::Int, censoring_upper::Int, data::AbstractVector{<:Real};
              n_samples::Int=1000, intercept_bounds::Union{Nothing, Tuple{Real,Real}}=nothing)

Perform neural posterior estimation for multiple systems estimation (MSE) parameters.

# Arguments
- `K`: Number of lists
- `censoring_lower`: Lower censoring bound
- `censoring_upper`: Upper censoring bound (use 0 for no censoring)
- `data`: Vector of observed counts (length should be 2^K - 1). Negative values indicate censored observations.
- `n_samples`: Number of posterior samples to draw (default: 1000)
- `intercept_bounds`: Optional tuple (lower, upper) for rejection sampling on the intercept parameter

# Returns
A matrix of posterior samples with dimensions (n_params, n_samples).
Parameters are ordered as: intercept, beta_1, ..., beta_K, gamma_12, gamma_13, ..., gamma_{K-1,K}

# Example
```julia
K = 5
data = [10, 5, 3, 8, 2, ...]  # 2^5 - 1 = 31 observations
samples = infer_npe(K, 0, 0, data; n_samples=1000)
# Apply prior bounds on intercept (e.g., Uniform(1, 10))
samples = infer_npe(K, 0, 0, data; n_samples=1000, intercept_bounds=(1, 10))
```
"""
function infer_npe(K::Int, censoring_lower::Int, censoring_upper::Int, data::AbstractVector{<:Real};
                   n_samples::Int=1000, intercept_bounds::Union{Nothing, Tuple{Real,Real}}=nothing)
    # Validate input size
    expected_size = 2^K - 1
    if length(data) != expected_size
        error("Expected $expected_size observations for K=$K lists, got $(length(data))")
    end

    # Prepare data (include censoring indicator if model expects it)
    X = prepare_data(data; censoring=(censoring_upper > 0))

    # Load estimator
    estimator = _load_npe_estimator(K, censoring_lower, censoring_upper)

    # Sample from posterior
    samples = sampleposterior(estimator, X, n_samples)

    # Apply rejection sampling on intercept if bounds provided
    if intercept_bounds !== nothing
        lower, upper = intercept_bounds
        mask = (samples[1, :] .>= lower) .& (samples[1, :] .<= upper)
        samples = samples[:, mask]
    end

    return samples
end
