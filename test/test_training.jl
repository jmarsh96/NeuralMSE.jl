using NeuralMSE
using Test
using Distributions
using Combinatorics

@testset "Model I/O" begin

    @testset "ModelConfig" begin
        # Test NBE config
        config_nbe = ModelConfig(
            model_type=:nbe,
            K=5,
            width=64,
            n_hidden=2,
            intercept_prior=(type=:Uniform, a=1.0, b=10.0),
            beta_prior=(type=:Normal, μ=0.0, σ=4.0),
            gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
        )

        @test config_nbe.model_type == :nbe
        @test config_nbe.K == 5
        @test config_nbe.width == 64
        @test config_nbe.n_hidden == 2
        @test config_nbe.encoding_dim === nothing
        @test config_nbe.censoring_lower == 0
        @test config_nbe.censoring_upper == 0

        # Test NPE config
        config_npe = ModelConfig(
            model_type=:npe,
            K=4,
            width=128,
            n_hidden=3,
            encoding_dim=64,
            intercept_prior=(type=:Uniform, a=1.0, b=10.0),
            beta_prior=(type=:Normal, μ=0.0, σ=4.0),
            gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
        )

        @test config_npe.model_type == :npe
        @test config_npe.encoding_dim == 64
    end

    @testset "save_model and load_model for NPE" begin
        mktempdir() do tmpdir
            # Test with a simple "estimator" (just for testing serialization)
            test_estimator = [1.0, 2.0, 3.0]

            config = ModelConfig(
                model_type=:npe,
                K=3,
                width=32,
                n_hidden=1,
                encoding_dim=16,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )

            # Test save
            model_id = save_model(tmpdir, test_estimator, config)
            @test model_id == 1

            # Test load by ID
            loaded_est, loaded_config = load_model(tmpdir, model_id)
            @test loaded_est == test_estimator
            @test loaded_config.K == 3
            @test loaded_config.model_type == :npe

            # Test load by config match
            loaded_est2, loaded_config2 = load_model(tmpdir;
                K=3, model_type=:npe
            )
            @test loaded_est2 == test_estimator
        end
    end

    @testset "save_nbe_model and load_model for NBE" begin
        mktempdir() do tmpdir
            # Test with simple "estimators"
            test_point = [1.0, 2.0, 3.0]
            test_interval = [4.0, 5.0, 6.0]

            config = ModelConfig(
                model_type=:nbe,
                K=3,
                width=32,
                n_hidden=1,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )

            # Test save
            model_id = save_nbe_model(tmpdir, test_point, test_interval, config)
            @test model_id == 1

            # Test load by ID - returns tuple of estimators
            (loaded_point, loaded_interval), loaded_config = load_model(tmpdir, model_id)
            @test loaded_point == test_point
            @test loaded_interval == test_interval
            @test loaded_config.K == 3
            @test loaded_config.model_type == :nbe

            # Test load by config match
            (loaded_point2, loaded_interval2), _ = load_model(tmpdir;
                K=3, model_type=:nbe
            )
            @test loaded_point2 == test_point
            @test loaded_interval2 == test_interval
        end
    end

    @testset "list_models and model management" begin
        mktempdir() do tmpdir
            # Save an NBE model
            config_nbe = ModelConfig(
                model_type=:nbe,
                K=3,
                width=32,
                n_hidden=1,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )
            nbe_id = save_nbe_model(tmpdir, [1.0], [2.0], config_nbe)

            # Save an NPE model
            config_npe = ModelConfig(
                model_type=:npe,
                K=3,
                width=32,
                n_hidden=1,
                encoding_dim=16,
                intercept_prior=(type=:Uniform, a=1.0, b=10.0),
                beta_prior=(type=:Normal, μ=0.0, σ=4.0),
                gamma_prior=(type=:Normal, μ=0.0, σ=4.0)
            )
            npe_id = save_model(tmpdir, [3.0], config_npe)

            # Test list_models
            models_df = list_models(tmpdir)
            @test nrow(models_df) == 2
            @test nbe_id in models_df.id
            @test npe_id in models_df.id

            # Test find_model
            found_nbe = find_model(tmpdir; K=3, model_type=:nbe)
            found_npe = find_model(tmpdir; K=3, model_type=:npe)
            @test found_nbe == nbe_id
            @test found_npe == npe_id

            # Test model_exists
            @test model_exists(tmpdir; K=3, model_type=:nbe)
            @test model_exists(tmpdir; K=3, model_type=:npe)
            @test !model_exists(tmpdir; K=5, model_type=:nbe)

            # Test remove_model
            remove_model(tmpdir, nbe_id)
            @test !model_exists(tmpdir; K=3, model_type=:nbe)
            @test model_exists(tmpdir; K=3, model_type=:npe)
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

            # Check model was saved (single NBE model now)
            models_df = list_models(tmpdir)
            @test nrow(models_df) == 1
            @test models_df.model_type[1] == :nbe

            # Load and verify
            (loaded_point, loaded_ci), config = load_model(tmpdir; K=3, model_type=:nbe)

            # Test inference produces same results
            data = rand(Float32, 2^K - 1, 1)
            @test point_est(data) ≈ loaded_point(data)
            @test ci_est(data) ≈ loaded_ci(data)
        end
    end

end
=#
