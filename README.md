# NeuralMSE

[![Build Status](https://github.com/jmarsh96/NeuralMSE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jmarsh96/NeuralMSE.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Neural network-based inference for Multiple Systems Estimation (MSE). This package provides pre-trained neural estimators for estimating population size and model parameters from capture-recapture data with multiple overlapping lists.

This package builds upon [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl), a general-purpose Julia framework for neural likelihood-free inference.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jmarsh96/NeuralMSE.jl")
```

On first use, pretrained models will be automatically downloaded (~400MB).

## Overview

NeuralMSE provides:

- **Neural Bayes Estimation (NBE)**: Fast point estimates with credible intervals
- **Neural Posterior Estimation (NPE)**: Full posterior samples via normalizing flows
- **Data Simulation**: Generate synthetic MSE data for research and validation
- **Model Training**: Train custom neural estimators with your own priors

All inference methods are amortized, meaning inference is nearly instantaneous once models are loaded.

## Quick Start

### Inference with Pretrained Models

```julia
using NeuralMSE

# Your capture-recapture data (counts for each list combination)
# For K=5 lists, you need 2^5 - 1 = 31 observations
K = 5
data = [45, 23, 12, 8, 15, 5, 3, ...]  # 31 values

# Neural Bayes Estimation - get point estimates and 95% credible intervals
result = infer_nbe(K, 0, 0, data)
println("Median estimates: ", result.median)
println("95% CI lower: ", result.ci_lower)
println("95% CI upper: ", result.ci_upper)

# Neural Posterior Estimation - get full posterior samples
samples = infer_npe(K, 0, 0, data; n_samples=1000)
```

### Simulating Data

```julia
using NeuralMSE
using Distributions

# Sample parameters from priors
K = 5
θ = sample_parameters(K)  # Default: Uniform(1,10) intercept, Normal(0,4) effects

# Simulate data
data = simulate_data(θ, 1)  # Single replicate

# Generate training datasets
θ_train, Z_train = simulate_dataset(K, 10000)  # 10,000 samples

# Custom priors
θ_custom = sample_parameters(K;
    intercept_dist=Uniform(2, 8),
    beta_dist=Normal(0, 2),
    gamma_dist=Normal(0, 1)
)
```

### Training Custom Models

```julia
using NeuralMSE

# Train NBE (point estimator + interval estimator)
point_est, ci_est = train_nbe(5;
    width=256,
    n_hidden=3,
    epochs=100,
    savepath="my_models/"
)

# Train NPE (posterior estimator)
npe = train_npe(5;
    width=256,
    n_hidden=3,
    encoding_dim=128,
    epochs=200,
    savepath="my_models/"
)
```

## API Reference

### Inference Functions

#### `infer_nbe(K, censoring_lower, censoring_upper, data)`

Perform Neural Bayes Estimation to obtain point estimates and credible intervals.

**Arguments:**
- `K::Int`: Number of lists (supported: 3-6, 10)
- `censoring_lower::Int`: Lower censoring bound (use 0 for no censoring)
- `censoring_upper::Int`: Upper censoring bound (use 0 for no censoring)
- `data::AbstractVector`: Vector of observed counts (length must be `2^K - 1`)

**Returns:** A named tuple with:
- `median`: Posterior median estimates for all parameters
- `ci_lower`: 2.5% quantile (lower bound of 95% credible interval)
- `ci_upper`: 97.5% quantile (upper bound of 95% credible interval)

#### `infer_npe(K, censoring_lower, censoring_upper, data; n_samples=1000, intercept_bounds=nothing)`

Perform Neural Posterior Estimation to obtain samples from the posterior distribution.

**Arguments:**
- `K::Int`: Number of lists
- `censoring_lower::Int`, `censoring_upper::Int`: Censoring bounds
- `data::AbstractVector`: Vector of observed counts
- `n_samples::Int`: Number of posterior samples (default: 1000)
- `intercept_bounds`: Optional tuple `(lower, upper)` for rejection sampling

**Returns:** A matrix of posterior samples with shape `(n_params, n_samples)`.

### Simulation Functions

#### `sample_parameters(K; intercept_dist, beta_dist, gamma_dist)`

Sample parameters from prior distributions.

```julia
θ = sample_parameters(5)  # 16 parameters for K=5
θ = sample_parameters(5; intercept_dist=Uniform(2, 8))
```

#### `simulate_data(θ, m; censoring_lower=0, censoring_upper=0)`

Simulate MSE capture-recapture data.

```julia
data = simulate_data(θ, 100)  # 100 replicates
data_censored = simulate_data(θ, 100; censoring_lower=1, censoring_upper=10)
```

#### `simulate_dataset(K, n_samples; m=1, priors...)`

Generate complete training datasets.

```julia
θ_train, Z_train = simulate_dataset(5, 10000)
```

### Training Functions

#### `train_nbe(K; kwargs...)`

Train Neural Bayes Estimators.

**Key Arguments:**
- `width`: MLP hidden layer width (default: 256)
- `n_hidden`: Number of hidden layers (default: 3)
- `epochs`: Training epochs (default: 100)
- `savepath`: Directory to save models (optional)

**Returns:** `(PointEstimator, IntervalEstimator)`

#### `train_npe(K; kwargs...)`

Train Neural Posterior Estimator.

**Additional Arguments:**
- `encoding_dim`: Summary statistic dimension (default: 128)
- `epochs`: Training epochs (default: 200)

**Returns:** `PosteriorEstimator`

### Model Management

#### `list_available_models()`

List all available pretrained models.

```julia
models = list_available_models()
filter(row -> row.K == 5 && row.model_type == :npe, models)
```

#### `save_model(path, estimator, config)`

Save a trained model with its configuration.

#### `load_model(path, model_id)` / `load_model(path; K, model_type, ...)`

Load a model by ID or by configuration match.

## Model Parameters

The estimators return parameters for a log-linear model:

```
log(λ_S) = α + Σᵢ βᵢ I(i ∈ S) + Σᵢ<ⱼ γᵢⱼ I(i ∈ S) I(j ∈ S)
```

**Parameter ordering:**
1. Intercept α (1 parameter)
2. Main effects β₁, β₂, ..., βₖ (K parameters)
3. Interactions γ₁₂, γ₁₃, ..., γ₍ₖ₋₁₎ₖ (K(K-1)/2 parameters)

Total: `1 + K + K(K-1)/2` parameters

## Handling Censored Data

When some cell counts are censored, indicate censored values with `-1`:

```julia
data = [45, 23, -1, 8, 15, -1, 3, ...]  # -1 means censored
result = infer_nbe(K, 0, 10, data)  # counts ≤10 were censored
```

## Input Data Format

Data should contain counts for each list combination, ordered by binary representation:

For K=3 lists (A, B, C):
| Index | Lists | Binary |
|-------|-------|--------|
| 1 | A only | 001 |
| 2 | B only | 010 |
| 3 | A+B | 011 |
| 4 | C only | 100 |
| 5 | A+C | 101 |
| 6 | B+C | 110 |
| 7 | A+B+C | 111 |

## Citation

If you use this package in your research, please cite:

```bibtex
@article{marsh2025neural,
  title={Neural Methods for Multiple Systems Estimation Models},
  author={Joseph Marsh and Nathan A. Judd and Lax Chan and Rowland G. Seymour},
  url={https://arxiv.org/abs/2601.05859},
  year={2026}
}
```

Additionally, please cite the underlying NeuralEstimators framework:

```bibtex
@Article{,
    title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
    author = {Matthew Sainsbury-Dale and Andrew Zammit-Mangion and Raphael Huser},
    journal = {The American Statistician},
    year = {2024},
    volume = {78},
    pages = {1--14},
    doi = {10.1080/00031305.2023.2249522},
}
```

## License

MIT License
