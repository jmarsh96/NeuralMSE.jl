#=
Data simulation functions for Multiple Systems Estimation (MSE) models.

This module provides functions for:
- Sampling parameters from prior distributions
- Simulating MSE capture-recapture data
- Generating training datasets for neural estimators
=#

using Distributions
using Combinatorics

#=
Internal helper functions
=#

"""
    _enumerate_all_combinations(K::Int) -> Vector{String}

Enumerate all non-empty subsets of K lists as comma-separated strings.

For K=3, returns: ["1", "2", "3", "1,2", "1,3", "2,3", "1,2,3"]
"""
function _enumerate_all_combinations(K::Int)
    combos = String[]
    for n in 1:K
        for c in combinations(1:K, n)
            push!(combos, join(c, ","))
        end
    end
    return combos
end

"""
    _enumerate_two_digit_numbers(K::Int) -> Vector{Vector{Int64}}

Enumerate all pairs of list indices for gamma (interaction) parameters.

For K=3, returns: [[1,2], [1,3], [2,3]]
"""
function _enumerate_two_digit_numbers(K::Int)
    numbers = Vector{Int64}[]
    for i in 1:K-1
        for j in i+1:K
            push!(numbers, [i, j])
        end
    end
    return numbers
end

"""
    _compute_digit_pairs(n::String) -> Tuple{Vector{Int}, Vector{Vector{Int}}}

Parse a combination string and return the individual digits and all pairs.

# Example
```julia
_compute_digit_pairs("1,2,3")  # Returns ([1,2,3], [[1,2], [1,3], [2,3]])
_compute_digit_pairs("1")      # Returns ([1], [])
```
"""
function _compute_digit_pairs(n::String)
    n_split = split(n, ",")
    if length(n_split) == 1
        return [parse(Int, n)], Vector{Int}[]
    end
    digits = [parse(Int, d) for d in n_split]
    pairs = collect(combinations(digits, 2))
    return digits, pairs
end

"""
    _rpois(logλ; logλ_max=43.0) -> Int

Custom Poisson sampler that handles overflow for large lambda values.

When logλ exceeds logλ_max (approximately where Float64 overflows),
samples from Poisson(exp(logλ_max)) instead.
"""
function _rpois(logλ; logλ_max=43.0)
    if logλ > logλ_max
        return rand(Poisson(exp(logλ_max)))
    else
        return rand(Poisson(exp(logλ)))
    end
end

#=
Public API functions
=#

"""
    sample_parameters(K::Int; intercept_dist=Uniform(1, 10),
                      beta_dist=Normal(0, 4), gamma_dist=Normal(0, 4)) -> Vector{Float32}

Sample parameters from prior distributions for the log-linear MSE model.

The log-linear model is:
    log(λ_S) = α + Σᵢ βᵢ I(i ∈ S) + Σᵢ<ⱼ γᵢⱼ I(i ∈ S) I(j ∈ S)

# Arguments
- `K::Int`: Number of lists
- `intercept_dist`: Distribution for the intercept parameter α (default: Uniform(1, 10))
- `beta_dist`: Distribution for main effect parameters βᵢ (default: Normal(0, 4))
- `gamma_dist`: Distribution for interaction parameters γᵢⱼ (default: Normal(0, 4))

# Returns
A Float32 vector of length 1 + K + binomial(K, 2) containing:
- Position 1: intercept (α)
- Positions 2 to K+1: main effects (β₁, ..., βₖ)
- Positions K+2 to end: interactions (γ₁₂, γ₁₃, ..., γₖ₋₁,ₖ)

# Example
```julia
params = sample_parameters(5)  # 16 parameters for K=5
params = sample_parameters(5; intercept_dist=Uniform(2, 8))  # Custom intercept prior
```
"""
function sample_parameters(
    K::Int;
    intercept_dist=Uniform(1, 10),
    beta_dist=Normal(0, 4),
    gamma_dist=Normal(0, 4)
)
    intercept = rand(intercept_dist)
    betas = rand(beta_dist, K)
    gammas = rand(gamma_dist, binomial(K, 2))
    return Float32.(vcat(intercept, betas, gammas))
end

"""
    simulate_data(θ::AbstractVector, m::Int;
                  censoring_lower::Int=0, censoring_upper::Int=0,
                  log_transform::Bool=true) -> Matrix{Float32}

Simulate MSE capture-recapture data from the log-linear model.

# Arguments
- `θ`: Parameter vector from `sample_parameters` (intercept, betas, gammas)
- `m`: Number of independent replicates to simulate
- `censoring_lower`: Lower censoring bound (counts in [lower, upper] are censored)
- `censoring_upper`: Upper censoring bound (use 0 for no censoring)
- `log_transform`: If true, return log(count + 1) transformed data (default: true)

# Returns
- Without censoring: Matrix of shape (2^K - 1, m) with (optionally log-transformed) counts
- With censoring (censoring_upper > 0): Matrix of shape (2*(2^K - 1), m) where:
  - First half: log-transformed counts or -1.0 for censored observations
  - Second half: censoring indicator (1 if censored, 0 otherwise)

# Example
```julia
K = 5
θ = sample_parameters(K)
data = simulate_data(θ, 100)  # 100 replicates, shape (31, 100)
data_censored = simulate_data(θ, 100; censoring_lower=1, censoring_upper=10)  # shape (62, 100)
```
"""
function simulate_data(
    θ::AbstractVector,
    m::Int;
    censoring_lower::Int=0,
    censoring_upper::Int=0,
    log_transform::Bool=true
)
    # Infer K from parameter vector length: len = 1 + K + K(K-1)/2
    # Solving: K = (-1 + sqrt(1 + 8*(len-1))) / 2
    K = Int(-0.5 + (sqrt(8 * length(θ) - 7) / 2))

    intercept = θ[1]
    betas = θ[2:1+K]
    gammas = θ[2+K:end]

    # Pre-compute list combinations and gamma mapping
    all_combinations = _enumerate_all_combinations(K)
    two_digit_numbers = _enumerate_two_digit_numbers(K)
    γ_map = Dict(two_digit_numbers .=> collect(1:length(gammas)))

    Z = zeros(2^K - 1, m)

    for j in 1:m
        for (i, list) in enumerate(all_combinations)
            logλ = intercept
            digits, digit_pairs = _compute_digit_pairs(list)

            # Add main effects
            for digit in digits
                logλ += betas[digit]
            end

            # Add interaction effects
            for pair in digit_pairs
                logλ += gammas[γ_map[pair]]
            end

            Z[i, j] = _rpois(logλ)
        end
    end

    # Handle censoring
    if censoring_upper > 0
        W = Float32.(1 * (censoring_lower .<= Z .<= censoring_upper))
        U = ifelse.(censoring_lower .<= Z .<= censoring_upper, -1.0f0, Float32.(log.(Z .+ 1)))
        return vcat(U, W)
    end

    # Return log-transformed or raw counts
    return log_transform ? Float32.(log.(Z .+ 1)) : Float32.(Z)
end

"""
    simulate_dataset(K::Int, n_samples::Int;
                     m::Int=1,
                     censoring_lower::Int=0,
                     censoring_upper::Int=0,
                     intercept_dist=Uniform(1, 10),
                     beta_dist=Normal(0, 4),
                     gamma_dist=Normal(0, 4)) -> Tuple{Matrix{Float32}, Matrix{Float32}}

Generate a complete training/validation dataset for neural estimator training.

This is a convenience function that combines `sample_parameters` and `simulate_data`
to generate paired (parameters, data) samples for training.

# Arguments
- `K`: Number of lists
- `n_samples`: Number of parameter/data pairs to generate
- `m`: Number of data replicates per parameter set (default: 1)
- `censoring_lower`, `censoring_upper`: Censoring bounds (default: no censoring)
- `intercept_dist`, `beta_dist`, `gamma_dist`: Prior distributions

# Returns
A tuple (θ, Z) where:
- θ: Parameter matrix of shape (n_params, n_samples) where n_params = 1 + K + binomial(K, 2)
- Z: Data matrix of shape (n_features, n_samples) where n_features = 2^K - 1 (or doubled with censoring)

# Example
```julia
θ_train, Z_train = simulate_dataset(5, 10000)  # 10,000 training samples
θ_val, Z_val = simulate_dataset(5, 1000; censoring_lower=1, censoring_upper=10)
```
"""
function simulate_dataset(
    K::Int,
    n_samples::Int;
    m::Int=1,
    censoring_lower::Int=0,
    censoring_upper::Int=0,
    intercept_dist=Uniform(1, 10),
    beta_dist=Normal(0, 4),
    gamma_dist=Normal(0, 4)
)
    n_params = 1 + K + binomial(K, 2)
    n_data = 2^K - 1
    if censoring_upper > 0
        n_data *= 2
    end

    θ = Matrix{Float32}(undef, n_params, n_samples)
    Z = Matrix{Float32}(undef, n_data, n_samples)

    for i in 1:n_samples
        params = sample_parameters(K;
            intercept_dist=intercept_dist,
            beta_dist=beta_dist,
            gamma_dist=gamma_dist
        )
        θ[:, i] = params

        data = simulate_data(params, m;
            censoring_lower=censoring_lower,
            censoring_upper=censoring_upper
        )
        Z[:, i] = vec(data)
    end

    return θ, Z
end

"""
    get_n_params(K::Int) -> Int

Return the number of parameters for a K-list MSE model.

Parameters = 1 (intercept) + K (main effects) + binomial(K,2) (interactions)
"""
get_n_params(K::Int) = 1 + K + binomial(K, 2)

"""
    get_n_data(K::Int; censoring::Bool=false) -> Int

Return the data dimension for a K-list MSE model.

Data dimension = 2^K - 1 (all non-empty list combinations)
Doubled if censoring is used (data + censoring indicators).
"""
get_n_data(K::Int; censoring::Bool=false) = censoring ? 2 * (2^K - 1) : 2^K - 1

"""
    get_param_names(K::Int) -> Vector{String}

Return parameter names for a K-list MSE model.

# Example
```julia
get_param_names(3)  # ["intercept", "beta_1", "beta_2", "beta_3", "gamma_1,2", "gamma_1,3", "gamma_2,3"]
```
"""
function get_param_names(K::Int)
    return vcat(
        "intercept",
        ["beta_$i" for i in 1:K],
        ["gamma_$i,$j" for i in 1:(K-1) for j in (i+1):K]
    )
end
