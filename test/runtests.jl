using NeuralMSE
using Test
using Combinatorics
using DataFrames

# Include test files
include("test_simulation.jl")
include("test_training.jl")

@testset "NeuralMSE.jl" begin

    @testset "prepare_data" begin
        # Test without censoring
        data = [10, 5, 0, 3]
        prepared = prepare_data(data)
        @test prepared isa Matrix{Float32}
        @test size(prepared) == (4, 1)

        # Test log transformation
        @test prepared[1, 1] ≈ log(11)  # log(10 + 1)
        @test prepared[2, 1] ≈ log(6)   # log(5 + 1)
        @test prepared[3, 1] ≈ 0.0f0    # log(0 + 1) = 0, but 0 stays 0

        # Test with censoring flag enabled
        data_censored = [10, -1, 5, -1]
        prepared_censored = prepare_data(data_censored; censoring=true)
        @test size(prepared_censored) == (8, 1)  # original + indicator

        # Check censoring indicator
        @test prepared_censored[6, 1] == 1.0  # second element was censored
        @test prepared_censored[8, 1] == 1.0  # fourth element was censored
        @test prepared_censored[5, 1] == 0.0  # first element was not censored
    end

    # Note: Inference tests require pretrained models to be available via artifacts.
    # These tests are skipped if the artifact is not available.
    # Once the Artifacts.toml is set up with the converted models, these tests will run.

    #=
    @testset "infer_nbe" begin
        K = 5
        n_obs = 2^K - 1  # 31 observations
        test_data = rand(1:20, n_obs)

        result = infer_nbe(K, 0, 0, test_data)

        @test haskey(result, :median)
        @test haskey(result, :ci_lower)
        @test haskey(result, :ci_upper)

        # Check dimensions
        n_params = 1 + K + binomial(K, 2)
        @test length(result.median) == n_params
        @test length(result.ci_lower) == n_params
        @test length(result.ci_upper) == n_params
    end

    @testset "infer_npe" begin
        K = 5
        n_obs = 2^K - 1
        test_data = rand(1:20, n_obs)

        samples = infer_npe(K, 0, 0, test_data; n_samples=100)

        n_params = 1 + K + binomial(K, 2)
        @test size(samples, 1) == n_params
        @test size(samples, 2) <= 100  # May be less due to rejection sampling

        # Test with intercept bounds
        samples_bounded = infer_npe(K, 0, 0, test_data; n_samples=100, intercept_bounds=(1.0, 10.0))
        @test all(samples_bounded[1, :] .>= 1.0)
        @test all(samples_bounded[1, :] .<= 10.0)
    end
    =#

end
