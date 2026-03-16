module NeuralMSE

#=
NeuralMSE.jl - Neural Estimation for Multiple Systems Estimation Models

This package provides neural network-based inference for log-linear MSE models,
including:
- Neural Bayes Estimation (NBE) for point estimates and credible intervals
- Neural Posterior Estimation (NPE) for full posterior distributions
- Data simulation for model training
- Training utilities for custom models
=#

# Core dependencies
using JLD2
using NeuralEstimators
using Distributions
using Combinatorics
using Folds
using Flux
using Dates
using DataFrames
# Note: Artifacts are loaded at runtime via Pkg.Artifacts in artifacts.jl
# to avoid compile-time errors when Artifacts.toml has placeholder values

import NeuralEstimators: sampleposterior

# Inference API (public)
export infer_nbe, infer_npe, prepare_data

# Simulation API (public)
export sample_parameters, simulate_data, simulate_dataset
export get_n_params, get_n_data, get_param_names

# Training API (public)
export train_nbe, train_npe

# Model I/O API (public)
export save_model, save_nbe_model, load_model, ModelConfig
export list_models, remove_model, model_exists
export load_model_registry, find_model

# Pretrained model management (public)
export get_models_path, list_available_models
export load_pretrained_model, find_pretrained_model
export pretrained_model_exists, print_available_models


include("simulation.jl")
include("model_io.jl")
include("artifacts.jl")
include("training.jl")
include("functions.jl")

end
