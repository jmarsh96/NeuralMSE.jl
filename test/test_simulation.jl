using NeuralMSE
using Test
using Distributions
using Combinatorics

@testset "Simulation API" begin

    @testset "sample_parameters" begin
        # Test default priors
        K = 5
        params = sample_parameters(K)
        n_expected = 1 + K + binomial(K, 2)  # 1 + 5 + 10 = 16

        @test length(params) == n_expected
        @test eltype(params) == Float32

        # Test that intercept is in expected range (Uniform(1, 10))
        # Sample multiple times to check bounds
        for _ in 1:10
            p = sample_parameters(K)
            @test 1.0 <= p[1] <= 10.0
        end

        # Test custom priors
        params_custom = sample_parameters(3;
            intercept_dist=Uniform(2, 5),
            beta_dist=Normal(0, 1),
            gamma_dist=Normal(0, 0.5)
        )
        n_expected_3 = 1 + 3 + binomial(3, 2)  # 1 + 3 + 3 = 7
        @test length(params_custom) == n_expected_3
    end

    @testset "simulate_data - basic" begin
        K = 4
        θ = sample_parameters(K)
        m = 10

        # Test basic simulation without censoring
        data = simulate_data(θ, m)
        @test size(data) == (2^K - 1, m)
        @test eltype(data) == Float32

        # All values should be non-negative (log-transformed counts)
        @test all(data .>= 0)

        # Test single replicate
        data_single = simulate_data(θ, 1)
        @test size(data_single) == (2^K - 1, 1)
    end

    @testset "simulate_data - censoring" begin
        K = 4
        θ = sample_parameters(K)
        m = 50

        # Test with censoring
        data_censored = simulate_data(θ, m; censoring_lower=1, censoring_upper=10)

        # With censoring, output should be doubled (U and W concatenated)
        @test size(data_censored) == (2 * (2^K - 1), m)

        # First half should contain either log-transformed values or -1 (censored)
        U = data_censored[1:2^K-1, :]
        W = data_censored[2^K:end, :]

        # Check censoring indicator is binary
        @test all(w -> w == 0.0 || w == 1.0, W)

        # Where W=1 (censored), U should be -1
        for i in 1:size(U, 1), j in 1:size(U, 2)
            if W[i, j] == 1.0
                @test U[i, j] == -1.0f0
            end
        end
    end

    @testset "simulate_data - no log transform" begin
        K = 3
        θ = sample_parameters(K)
        m = 5

        # Test raw counts (no log transform)
        data_raw = simulate_data(θ, m; log_transform=false)
        @test size(data_raw) == (2^K - 1, m)

        # Raw counts should be non-negative integers (stored as Float32)
        @test all(d -> d >= 0 && d == floor(d), data_raw)
    end

    @testset "simulate_dataset" begin
        K = 4
        n_samples = 100

        θ, Z = simulate_dataset(K, n_samples)

        n_params = 1 + K + binomial(K, 2)
        n_data = 2^K - 1

        @test size(θ) == (n_params, n_samples)
        @test size(Z) == (n_data, n_samples)
        @test eltype(θ) == Float32
        @test eltype(Z) == Float32

        # Test with censoring
        θ_c, Z_c = simulate_dataset(K, n_samples; censoring_lower=1, censoring_upper=5)
        @test size(Z_c) == (2 * n_data, n_samples)
    end

    @testset "utility functions" begin
        @test get_n_params(3) == 1 + 3 + 3  # 7
        @test get_n_params(5) == 1 + 5 + 10  # 16

        @test get_n_data(3) == 7  # 2^3 - 1
        @test get_n_data(5) == 31  # 2^5 - 1
        @test get_n_data(3; censoring=true) == 14  # doubled

        param_names = get_param_names(3)
        @test length(param_names) == 7
        @test param_names[1] == "intercept"
        @test param_names[2] == "beta_1"
        @test param_names[5] == "gamma_1,2"
    end

end
