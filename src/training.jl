#=
Training functions for Neural Bayes Estimators (NBE) and Neural Posterior Estimators (NPE).

This module provides:
- MLP network construction for MSE models
- train_nbe: Train point and interval estimators
- train_npe: Train posterior estimators using normalizing flows
=#

using Flux
using Distributions
using Combinatorics
using Folds
using NeuralEstimators

#=
Network construction
=#

"""
    _construct_mlp(width::Int, n_hidden::Int, K::Int;
                   censoring::Bool=false, intercept_support=nothing) -> Chain

Construct an MLP network for neural estimation.

# Arguments
- `width`: Width of hidden layers
- `n_hidden`: Number of hidden layers
- `K`: Number of lists
- `censoring`: Whether the model handles censored data (doubles input dimension)
- `intercept_support`: Optional tuple (a, b) to compress intercept output to [a, b]

# Returns
A Flux Chain representing the MLP architecture.
"""
function _construct_mlp(width::Int, n_hidden::Int, K::Int;
                        censoring::Bool=false, intercept_support=nothing)
    n_data = 2^K - 1
    n_pars = 1 + K + binomial(K, 2)  # intercept + betas + gammas

    if censoring
        n_data *= 2  # Double input size for censored data (U and W)
    end

    # Build hidden layers
    layers = Any[Dense(n_data, width, relu)]
    for _ in 1:n_hidden
        push!(layers, Dense(width, width, relu))
    end

    # Output layer with optional intercept compression
    if intercept_support !== nothing
        a, b = Float32(intercept_support[1]), Float32(intercept_support[2])
        final_layer = Parallel(
            vcat,
            Chain(Dense(width, 1, identity), Compress(a, b)),  # Compress intercept to prior support
            Dense(width, n_pars - 1, identity)  # Identity for betas and gammas
        )
    else
        final_layer = Dense(width, n_pars)
    end

    push!(layers, final_layer)
    return Chain(layers...)
end

"""
    _construct_encoder(width::Int, n_hidden::Int, K::Int, encoding_dim::Int;
                       censoring::Bool=false) -> Chain

Construct an encoder network for NPE.

# Arguments
- `width`: Width of hidden layers
- `n_hidden`: Number of hidden layers
- `K`: Number of lists
- `encoding_dim`: Output dimension of the encoder
- `censoring`: Whether the model handles censored data

# Returns
A Flux Chain representing the encoder architecture.
"""
function _construct_encoder(width::Int, n_hidden::Int, K::Int, encoding_dim::Int;
                            censoring::Bool=false)
    n_data = 2^K - 1
    if censoring
        n_data *= 2
    end

    layers = Any[Dense(n_data, width, relu)]
    for _ in 1:n_hidden
        push!(layers, Dense(width, width, relu))
    end
    push!(layers, Dense(width, encoding_dim))

    return Chain(layers...)
end

"""
    _dist_to_named_tuple(dist) -> NamedTuple

Convert a Distribution to a NamedTuple for serialization.
"""
function _dist_to_named_tuple(dist)
    if dist isa Uniform
        return (type=:Uniform, a=Float64(minimum(dist)), b=Float64(maximum(dist)))
    elseif dist isa Normal
        return (type=:Normal, μ=Float64(mean(dist)), σ=Float64(std(dist)))
    else
        error("Unsupported distribution type: $(typeof(dist))")
    end
end

#=
Training functions
=#

"""
    train_nbe(K::Int; kwargs...) -> Tuple{PointEstimator, IntervalEstimator}

Train Neural Bayes Estimators (NBE) for MSE parameter inference.

Returns both a point estimator (for median estimates) and an interval estimator
(for 95% credible intervals).

# Arguments
- `K::Int`: Number of lists

# Keyword Arguments
- `width::Int=256`: Width of MLP hidden layers
- `n_hidden::Int=3`: Number of hidden layers
- `train_size::Int=10000`: Number of training samples per epoch
- `m::Int=1`: Number of data replicates per parameter sample
- `epochs::Int=100`: Number of training epochs
- `censoring_lower::Int=0`: Lower censoring bound
- `censoring_upper::Int=0`: Upper censoring bound (0 = no censoring)
- `intercept_dist=Uniform(1, 10)`: Prior distribution for intercept
- `beta_dist=Normal(0, 4)`: Prior distribution for main effects
- `gamma_dist=Normal(0, 4)`: Prior distribution for interactions
- `savepath::Union{Nothing,String}=nothing`: If provided, save models to this directory
- `verbose::Bool=true`: Print training progress

# Returns
A tuple (point_estimator, interval_estimator).

# Example
```julia
# Train with default settings
point_est, ci_est = train_nbe(5)

# Train with custom architecture and save
point_est, ci_est = train_nbe(5;
    width=128, n_hidden=2,
    savepath="models/"
)

# Use the trained estimators
data = prepare_data(observed_counts)
median_estimates = point_est(data)
ci_bounds = ci_est(data)  # [lower; upper]
```
"""
function train_nbe(K::Int;
                   width::Int=256,
                   n_hidden::Int=3,
                   train_size::Int=10000,
                   m::Int=1,
                   epochs::Int=100,
                   censoring_lower::Int=0,
                   censoring_upper::Int=0,
                   intercept_dist=Uniform(1, 10),
                   beta_dist=Normal(0, 4),
                   gamma_dist=Normal(0, 4),
                   savepath::Union{Nothing,String}=nothing,
                   verbose::Bool=true)

    # Get intercept support for output compression (if using Uniform prior)
    intercept_support = intercept_dist isa Uniform ? params(intercept_dist) : nothing

    # Pre-compute list combinations
    all_combinations = _enumerate_all_combinations(K)
    two_digit_numbers = _enumerate_two_digit_numbers(K)

    # Define sampling function for parameters
    function sample_nbe(n_reps)
        return hcat([sample_parameters(K;
            intercept_dist=intercept_dist,
            beta_dist=beta_dist,
            gamma_dist=gamma_dist
        ) for _ in 1:n_reps]...)
    end

    # Define simulation function for data
    function simulate_nbe(θ, m_sim)
        Z = Folds.map(eachcol(θ)) do params
            _simulate_data_internal(params, m_sim;
                censoring_lower=censoring_lower,
                censoring_upper=censoring_upper,
                all_combinations=all_combinations,
                two_digit_numbers=two_digit_numbers
            )
        end
        return hcat(Z...)
    end

    # Construct network
    censoring = censoring_upper > 0
    network = _construct_mlp(width, n_hidden, K;
        censoring=censoring,
        intercept_support=intercept_support
    )

    # Create estimators
    point_estimator = PointEstimator(network)
    ci_estimator = IntervalEstimator(network)

    # Train point estimator
    if verbose
        println("Training point estimator...")
    end
    point_estimator = train(
        point_estimator,
        sample_nbe,
        simulate_nbe;
        K=train_size,
        m=m,
        epochs=epochs,
        verbose=verbose
    )

    # Train interval estimator
    if verbose
        println("Training interval estimator...")
    end
    ci_estimator = train(
        ci_estimator,
        sample_nbe,
        simulate_nbe;
        K=train_size,
        m=m,
        epochs=epochs,
        verbose=verbose
    )

    # Save if requested
    if savepath !== nothing
        config_point = ModelConfig(
            model_type=:nbe_point,
            K=K,
            width=width,
            n_hidden=n_hidden,
            train_size=train_size,
            m=m,
            censoring_lower=censoring_lower,
            censoring_upper=censoring_upper,
            intercept_prior=_dist_to_named_tuple(intercept_dist),
            beta_prior=_dist_to_named_tuple(beta_dist),
            gamma_prior=_dist_to_named_tuple(gamma_dist)
        )

        config_ci = ModelConfig(
            model_type=:nbe_interval,
            K=K,
            width=width,
            n_hidden=n_hidden,
            train_size=train_size,
            m=m,
            censoring_lower=censoring_lower,
            censoring_upper=censoring_upper,
            intercept_prior=_dist_to_named_tuple(intercept_dist),
            beta_prior=_dist_to_named_tuple(beta_dist),
            gamma_prior=_dist_to_named_tuple(gamma_dist)
        )

        point_id = save_model(savepath, point_estimator, config_point)
        ci_id = save_model(savepath, ci_estimator, config_ci)

        if verbose
            println("Saved point estimator with ID: $point_id")
            println("Saved interval estimator with ID: $ci_id")
        end
    end

    return point_estimator, ci_estimator
end

"""
    train_npe(K::Int; kwargs...) -> PosteriorEstimator

Train a Neural Posterior Estimator (NPE) for MSE parameter inference.

Uses normalizing flows to approximate the full posterior distribution,
enabling posterior sampling for uncertainty quantification.

# Arguments
- `K::Int`: Number of lists

# Keyword Arguments
- `width::Int=256`: Width of encoder hidden layers
- `n_hidden::Int=3`: Number of encoder hidden layers
- `encoding_dim::Int=128`: Dimension of the summary statistic encoding
- `train_size::Int=10000`: Number of training samples per epoch
- `m::Int=1`: Number of data replicates per parameter sample
- `epochs::Int=200`: Number of training epochs
- `censoring_lower::Int=0`: Lower censoring bound
- `censoring_upper::Int=0`: Upper censoring bound (0 = no censoring)
- `intercept_dist=Uniform(1, 10)`: Prior distribution for intercept
- `beta_dist=Normal(0, 4)`: Prior distribution for main effects
- `gamma_dist=Normal(0, 4)`: Prior distribution for interactions
- `savepath::Union{Nothing,String}=nothing`: If provided, save model to this directory
- `verbose::Bool=true`: Print training progress

# Returns
A PosteriorEstimator that can be used with `sampleposterior`.

# Example
```julia
# Train NPE
npe = train_npe(5; epochs=200)

# Sample from posterior
data = prepare_data(observed_counts)
samples = sampleposterior(npe, data, 1000)  # 1000 posterior samples
```
"""
function train_npe(K::Int;
                   width::Int=256,
                   n_hidden::Int=3,
                   encoding_dim::Int=128,
                   train_size::Int=10000,
                   m::Int=1,
                   epochs::Int=200,
                   censoring_lower::Int=0,
                   censoring_upper::Int=0,
                   intercept_dist=Uniform(1, 10),
                   beta_dist=Normal(0, 4),
                   gamma_dist=Normal(0, 4),
                   savepath::Union{Nothing,String}=nothing,
                   verbose::Bool=true)

    n_pars = 1 + K + binomial(K, 2)

    # Pre-compute list combinations
    all_combinations = _enumerate_all_combinations(K)
    two_digit_numbers = _enumerate_two_digit_numbers(K)

    # Define sampling function for parameters
    function sample_npe(n_reps)
        return hcat([sample_parameters(K;
            intercept_dist=intercept_dist,
            beta_dist=beta_dist,
            gamma_dist=gamma_dist
        ) for _ in 1:n_reps]...)
    end

    # Define simulation function for data
    function simulate_npe(θ, m_sim)
        Z = Folds.map(eachcol(θ)) do params
            _simulate_data_internal(params, m_sim;
                censoring_lower=censoring_lower,
                censoring_upper=censoring_upper,
                all_combinations=all_combinations,
                two_digit_numbers=two_digit_numbers
            )
        end
        return hcat(Z...)
    end

    # Construct encoder network
    censoring = censoring_upper > 0
    encoder = _construct_encoder(width, n_hidden, K, encoding_dim; censoring=censoring)

    # Create normalizing flow and posterior estimator
    flow = NormalisingFlow(n_pars, encoding_dim)
    estimator = PosteriorEstimator(flow, encoder)

    # Train
    if verbose
        println("Training NPE...")
    end
    estimator = train(
        estimator,
        sample_npe,
        simulate_npe;
        K=train_size,
        m=m,
        epochs=epochs,
        verbose=verbose
    )

    # Save if requested
    if savepath !== nothing
        config = ModelConfig(
            model_type=:npe,
            K=K,
            width=width,
            n_hidden=n_hidden,
            encoding_dim=encoding_dim,
            train_size=train_size,
            m=m,
            censoring_lower=censoring_lower,
            censoring_upper=censoring_upper,
            intercept_prior=_dist_to_named_tuple(intercept_dist),
            beta_prior=_dist_to_named_tuple(beta_dist),
            gamma_prior=_dist_to_named_tuple(gamma_dist)
        )

        model_id = save_model(savepath, estimator, config)

        if verbose
            println("Saved NPE with ID: $model_id")
        end
    end

    return estimator
end

#=
Internal simulation function (optimized version using pre-computed combinations)
=#

"""
Internal simulation function that accepts pre-computed combinations for efficiency.
"""
function _simulate_data_internal(
    θ::AbstractVector,
    m::Int;
    censoring_lower::Int=0,
    censoring_upper::Int=0,
    all_combinations::Vector{String},
    two_digit_numbers::Vector{Vector{Int64}}
)
    K = length(all_combinations) > 0 ? _infer_K_from_combinations(length(all_combinations)) : Int(-0.5 + (sqrt(8 * length(θ) - 7) / 2))

    intercept = θ[1]
    betas = θ[2:1+K]
    gammas = θ[2+K:end]

    γ_map = Dict(two_digit_numbers .=> collect(1:length(gammas)))

    Z = zeros(Float32, 2^K - 1, m)

    for j in 1:m
        for (i, list) in enumerate(all_combinations)
            logλ = intercept
            digits, digit_pairs = _compute_digit_pairs(list)

            for digit in digits
                logλ += betas[digit]
            end

            for pair in digit_pairs
                logλ += gammas[γ_map[pair]]
            end

            Z[i, j] = _rpois(logλ)
        end
    end

    if censoring_upper > 0
        W = Float32.(1 * (censoring_lower .<= Z .<= censoring_upper))
        U = ifelse.(censoring_lower .<= Z .<= censoring_upper, -1.0f0, Float32.(log.(Z .+ 1)))
        return vcat(U, W)
    end

    return Float32.(log.(Z .+ 1))
end

"""
Infer K from the number of combinations (2^K - 1).
"""
function _infer_K_from_combinations(n_combinations::Int)
    K = 1
    while 2^K - 1 < n_combinations
        K += 1
    end
    return K
end
