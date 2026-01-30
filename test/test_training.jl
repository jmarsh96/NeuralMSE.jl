using NeuralMSE
using Test
using Distributions
using Combinatorics

@testset "Model I/O" begin

    @testset "ModelConfig" begin
        config = ModelConfig(
            model_type=:nbe_point,
            K=5,
            width=64,
            n_hidden=2,
            intercept_prior=(type=:Uniform, a=1.0, b=10.0),
            beta_prior=(type=:Normal, μ=0.0, σ=4.0),
            gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
        )

        @test config.model_type == :nbe_point
        @test config.K == 5
        @test config.width == 64
        @test config.n_hidden == 2
        @test config.encoding_dim === nothing
        @test config.censoring_lower == 0
        @test config.censoring_upper == 0
    end

    @testset "save_model and load_model" begin
        mktempdir() do tmpdir
            # Create a simple test "estimator" (just a number for testing)
            # In real use this would be a PointEstimator, etc.
            test_estimator = [1.0, 2.0, 3.0]

            config = ModelConfig(
                model_type=:nbe_point,
                K=3,
                width=32,
                n_hidden=1,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )

            # Test save
            model_id = save_model(tmpdir, test_estimator, config)
            @test model_id == 1  # First model should get ID 1

            # Test load by ID
            loaded_est, loaded_config = load_model(tmpdir, model_id)
            @test loaded_est == test_estimator
            @test loaded_config.K == 3
            @test loaded_config.model_type == :nbe_point

            # Test load by config match
            loaded_est2, loaded_config2 = load_model(tmpdir;
                K=3, model_type=:nbe_point
            )
            @test loaded_est2 == test_estimator

            # Save another model
            config2 = ModelConfig(
                model_type=:nbe_interval,
                K=3,
                width=32,
                n_hidden=1,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )
            model_id2 = save_model(tmpdir, [4.0, 5.0, 6.0], config2)
            @test model_id2 == 2

            # Test list_models
            models_df = list_models(tmpdir)
            @test nrow(models_df) == 2
            @test 1 in models_df.id
            @test 2 in models_df.id

            # Test find_model
            found_id = find_model(tmpdir; K=3, model_type=:nbe_interval)
            @test found_id == 2

            # Test model_exists
            @test model_exists(tmpdir; K=3, model_type=:nbe_point)
            @test !model_exists(tmpdir; K=5, model_type=:nbe_point)

            # Test remove_model
            remove_model(tmpdir, model_id)
            @test !model_exists(tmpdir; K=3, model_type=:nbe_point)
            @test model_exists(tmpdir; K=3, model_type=:nbe_interval)
        end
    end

    @testset "load_model_registry" begin
        mktempdir() do tmpdir
            # Empty registry
            registry = load_model_registry(tmpdir)
            @test isempty(registry)

            # Add a model
            config = ModelConfig(
                model_type=:npe,
                K=4,
                width=64,
                n_hidden=2,
                encoding_dim=32,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )
            save_model(tmpdir, "test_npe", config)

            # Check registry
            registry = load_model_registry(tmpdir)
            @test length(registry) == 1
            @test haskey(registry, 1)
            @test registry[1].model_type == :npe
            @test registry[1].encoding_dim == 32
        end
    end

end

# Note: Training tests are commented out by default as they require
# NeuralEstimators to be fully loaded and can be slow.
# Uncomment to run full training tests.

#=
@testset "Training API" begin

    @testset "train_nbe - minimal" begin
        # Use very small architecture for fast testing
        K = 3
        point_est, ci_est = train_nbe(K;
            width=16,
            n_hidden=1,
            train_size=100,
            epochs=2,
            verbose=false
        )

        # Check types
        @test point_est isa NeuralEstimators.PointEstimator
        @test ci_est isa NeuralEstimators.IntervalEstimator

        # Test inference works
        data = rand(Float32, 2^K - 1, 1)
        result = point_est(data)
        n_params = 1 + K + binomial(K, 2)
        @test size(result, 1) == n_params
    end

    @testset "train_npe - minimal" begin
        K = 3
        npe = train_npe(K;
            width=16,
            n_hidden=1,
            encoding_dim=16,
            train_size=100,
            epochs=2,
            verbose=false
        )

        @test npe isa NeuralEstimators.PosteriorEstimator

        # Test sampling works
        data = rand(Float32, 2^K - 1, 1)
        samples = NeuralEstimators.sampleposterior(npe, data, 10)
        n_params = 1 + K + binomial(K, 2)
        @test size(samples, 1) == n_params
        @test size(samples, 2) == 10
    end

    @testset "train and save roundtrip" begin
        mktempdir() do tmpdir
            K = 3
            point_est, ci_est = train_nbe(K;
                width=16,
                n_hidden=1,
                train_size=100,
                epochs=2,
                savepath=tmpdir,
                verbose=false
            )

            # Check models were saved
            models_df = list_models(tmpdir)
            @test nrow(models_df) == 2  # point + interval

            # Load and verify
            loaded_point, config = load_model(tmpdir; K=3, model_type=:nbe_point)

            # Test inference produces same results
            data = rand(Float32, 2^K - 1, 1)
            @test point_est(data) ≈ loaded_point(data)
        end
    end

end
=#
