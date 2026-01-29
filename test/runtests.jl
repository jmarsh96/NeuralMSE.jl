using NeuralMSE
using Test

@testset "NeuralMSE.jl" begin
    # Test data preparation
    @testset "prepare_data" begin
        # Test without censoring
        data = [10, 5, 0, 3]
        prepared = prepare_data(data)
        @test prepared isa Matrix{Float32}
        @test size(prepared) == (4, 1)

        # Test with censoring flag enabled
        data_censored = [10, -1, 5, -1]
        prepared_censored = prepare_data(data_censored; censoring=true)
        @test size(prepared_censored) == (8, 1)  # original + indicator
    end

    # Test infer_nbe
    @testset "infer_nbe" begin
        K = 5
        n_obs = 2^K - 1  # 31 observations
        test_data = rand(1:20, n_obs)

        result = infer_nbe(K, 0, 2, test_data)

        @test haskey(result, :median)
        @test haskey(result, :ci_lower)
        @test haskey(result, :ci_upper)

        # Check dimensions
        n_params = 1 + K + binomial(K, 2)
        @test length(result.median) == n_params
        @test length(result.ci_lower) == n_params
        @test length(result.ci_upper) == n_params
    end

    # Test infer_npe
    @testset "infer_npe" begin
        K = 5
        n_obs = 2^K - 1
        test_data = rand(1:20, n_obs)

        samples = infer_npe(K, 0, 2, test_data; n_samples=100)

        n_params = 1 + K + binomial(K, 2)
        @test size(samples, 1) == n_params
        @test size(samples, 2) <= 100  # May be less due to rejection sampling

        # Test with intercept bounds
        samples_bounded = infer_npe(K, 0, 2, test_data; n_samples=100, intercept_bounds=(1.0, 10.0))
        @test all(samples_bounded[1, :] .>= 1.0)
        @test all(samples_bounded[1, :] .<= 10.0)
    end
end
