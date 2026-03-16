#=
Core inference functions for NeuralMSE.

This module provides the main public API functions:
- infer_nbe: Neural Bayes Estimation (point estimates + CIs)
- infer_npe: Neural Posterior Estimation (posterior sampling)
- prepare_data: Data preprocessing for neural estimators
=#

using NeuralEstimators
import NeuralEstimators: sampleposterior

#=
Internal model loading functions
=#

"""
Load NBE model (both point and interval estimators) from pretrained models.
"""
function _load_nbe_model(K::Int, censoring_lower::Int, censoring_upper::Int)
    models_path = get_models_path()
    (point_estimator, interval_estimator), _ = load_model(models_path;
        K=K,
        model_type=:nbe,
        censoring_lower=censoring_lower,
        censoring_upper=censoring_upper
    )
    return point_estimator, interval_estimator
end

"""
Load NPE estimator from pretrained models.
"""
function _load_npe_estimator(K::Int, censoring_lower::Int, censoring_upper::Int)
    models_path = get_models_path()
    estimator, _ = load_model(models_path;
        K=K,
        model_type=:npe,
        censoring_lower=censoring_lower,
        censoring_upper=censoring_upper
    )
    return estimator
end

#=
Data preparation
=#

"""
    prepare_data(data::AbstractVector{<:Real}; censoring::Union{Bool,Nothing}=nothing) -> Matrix{Float32}

Prepare input data for neural estimation.

Applies log transformation to positive counts and handles censored observations
(indicated by negative values).

# Arguments
- `data`: Vector of counts ordered by list combination. Negative values indicate censored observations.
- `censoring`: Controls censoring indicator output:
  - `nothing` (default): Auto-detect based on presence of negative values
  - `true`: Always append censoring indicator vector
  - `false`: Never append censoring indicator vector

# Returns
A Float32 matrix of shape (n_features, 1) suitable for input to neural estimators.

# Data ordering
For K lists, the data should have 2^K - 1 elements ordered by binary representation:
- K=3: indices 1-7 correspond to lists {1}, {2}, {1,2}, {3}, {1,3}, {2,3}, {1,2,3}
- More generally, index i corresponds to the list combination given by binary digits of i

# Example
```julia
# For K=3, observed counts for each list combination
counts = [10, 5, 3, 8, 2, 4, 1]  # 7 = 2^3 - 1 observations
X = prepare_data(counts)

# With censoring (negative values indicate censored) - auto-detected
counts_censored = [10, -1, 3, -1, 2, 4, 1]  # -1 means censored
X = prepare_data(counts_censored)  # Automatically detects censoring
```
"""
function prepare_data(data::AbstractVector{<:Real}; censoring::Union{Bool,Nothing}=nothing)
    # Auto-detect censoring if not specified
    has_censored = any(x -> x < 0, data)
    use_censoring = censoring === nothing ? has_censored : censoring

    # Log transform positive counts, keep negative (censored) as is
    out = [x > 0 ? log(x + 1) : Float32(x) for x in data]

    if use_censoring
        # Append censoring indicator vector (1 if censored, 0 otherwise)
        append!(out, Float32.(1 * (out .< 0)))
    end

    return reshape(Float32.(out), (length(out), 1))
end

#=
Public inference API
=#

"""
    infer_nbe(K::Int, censoring_lower::Int, censoring_upper::Int,
              data::AbstractVector{<:Real}) -> NamedTuple

Perform Neural Bayes Estimation for MSE model parameters.

Uses pretrained neural networks to provide point estimates (posterior medians)
and 95% credible intervals for the log-linear MSE model parameters.

# Arguments
- `K`: Number of lists
- `censoring_lower`: Lower censoring bound
- `censoring_upper`: Upper censoring bound (use 0 for no censoring)
- `data`: Vector of observed counts (length should be 2^K - 1). Negative values indicate censored observations.

# Returns
A named tuple with:
- `median`: Posterior median estimates for all parameters
- `ci_lower`: 2.5% quantile (lower bound of 95% CI)
- `ci_upper`: 97.5% quantile (upper bound of 95% CI)

Parameters are ordered as: intercept, β₁, ..., βₖ, γ₁₂, γ₁₃, ..., γₖ₋₁,ₖ

# Example
```julia
K = 5
data = [10, 5, 3, 8, 2, 4, 1, 6, 3, 2, 1, 5, 2, 1, 1,
        7, 4, 2, 1, 3, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]  # 31 observations

result = infer_nbe(K, 0, 0, data)
println("Intercept: \$(result.median[1]) [\$(result.ci_lower[1]), \$(result.ci_upper[1])]")
```
"""
function infer_nbe(K::Int, censoring_lower::Int, censoring_upper::Int,
                   data::AbstractVector{<:Real})
    # Validate input size
    expected_size = 2^K - 1
    if length(data) != expected_size
        error("Expected $expected_size observations for K=$K lists, got $(length(data))")
    end

    # Prepare data (include censoring indicator if model expects it)
    X = prepare_data(data; censoring=(censoring_upper > 0))

    # Load NBE model (contains both point and interval estimators)
    point_estimator, ci_estimator = _load_nbe_model(K, censoring_lower, censoring_upper)

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
    infer_npe(K::Int, censoring_lower::Int, censoring_upper::Int,
              data::AbstractVector{<:Real};
              n_samples::Int=1000,
              intercept_bounds::Union{Nothing, Tuple{Real,Real}}=nothing) -> Matrix{Float32}

Perform Neural Posterior Estimation for MSE model parameters.

Uses a pretrained normalizing flow to approximate the posterior distribution
and generate samples for uncertainty quantification.

# Arguments
- `K`: Number of lists
- `censoring_lower`: Lower censoring bound
- `censoring_upper`: Upper censoring bound (use 0 for no censoring)
- `data`: Vector of observed counts (length should be 2^K - 1). Negative values indicate censored observations.
- `n_samples`: Number of posterior samples to draw (default: 1000)
- `intercept_bounds`: Optional tuple (lower, upper) for rejection sampling on the intercept parameter. Use this to enforce prior bounds (e.g., `(1, 10)` for Uniform(1, 10) prior).

# Returns
A Float32 matrix of posterior samples with dimensions (n_params, n_samples).
Parameters are ordered as: intercept, β₁, ..., βₖ, γ₁₂, γ₁₃, ..., γₖ₋₁,ₖ

Note: When using `intercept_bounds`, the returned matrix may have fewer than
`n_samples` columns if many samples are rejected.

# Example
```julia
K = 5
data = [10, 5, 3, 8, 2, 4, 1, 6, 3, 2, 1, 5, 2, 1, 1,
        7, 4, 2, 1, 3, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

# Basic posterior sampling
samples = infer_npe(K, 0, 0, data; n_samples=1000)

# With prior bounds on intercept (Uniform(1, 10))
samples = infer_npe(K, 0, 0, data; n_samples=1000, intercept_bounds=(1, 10))

# Compute posterior summary
intercept_samples = samples[1, :]
println("Intercept: \$(median(intercept_samples)) [95% CI: \$(quantile(intercept_samples, 0.025)), \$(quantile(intercept_samples, 0.975))]")
```
"""
function infer_npe(K::Int, censoring_lower::Int, censoring_upper::Int,
                   data::AbstractVector{<:Real};
                   n_samples::Int=1000,
                   intercept_bounds::Union{Nothing, Tuple{Real,Real}}=nothing)
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
