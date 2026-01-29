# NeuralMSE

[![Build Status](https://github.com/jmarsh96/NeuralMSE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jmarsh96/NeuralMSE.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Neural network-based inference for Multiple Systems Estimation (MSE). This package provides pre-trained neural estimators for estimating population size and model parameters from capture-recapture data with multiple overlapping lists.

This package builds upon [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl), a general-purpose Julia framework for neural likelihood-free inference. For details on the methodology and its application to MSE, see references at the bottom.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jmarsh96/NeuralMSE.jl")
```

## Overview

NeuralMSE provides two estimation approaches:

- **Neural Bayes Estimation (NBE)**: Fast point estimates with credible intervals
- **Neural Posterior Estimation (NPE)**: Full posterior samples via normalizing flows

Both methods are amortized, meaning inference is nearly instantaneous once models are loaded.

## Quick Start

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

## API Reference

### `infer_nbe(K, censoring_lower, censoring_upper, data)`

Perform Neural Bayes Estimation to obtain point estimates and credible intervals.

**Arguments:**
- `K::Int`: Number of lists (supported: 3-6)
- `censoring_lower::Int`: Lower censoring bound (use 0 for no censoring)
- `censoring_upper::Int`: Upper censoring bound (use 0 for no censoring)
- `data::AbstractVector`: Vector of observed counts (length must be `2^K - 1`)

**Returns:** A named tuple with:
- `median`: Posterior median estimates for all parameters
- `ci_lower`: 2.5% quantile (lower bound of 95% credible interval)
- `ci_upper`: 97.5% quantile (upper bound of 95% credible interval)

**Example:**
```julia
K = 5
data = rand(1:50, 2^K - 1)  # 31 observations
result = infer_nbe(K, 0, 0, data)

# Access results
intercept = result.median[1]
betas = result.median[2:K+1]
gammas = result.median[K+2:end]
```

### `infer_npe(K, censoring_lower, censoring_upper, data; n_samples=1000, intercept_bounds=nothing)`

Perform Neural Posterior Estimation to obtain samples from the posterior distribution.

**Arguments:**
- `K::Int`: Number of lists (supported: 3-6)
- `censoring_lower::Int`: Lower censoring bound
- `censoring_upper::Int`: Upper censoring bound (use 0 for no censoring)
- `data::AbstractVector`: Vector of observed counts (length must be `2^K - 1`)
- `n_samples::Int`: Number of posterior samples to draw (default: 1000)
- `intercept_bounds::Union{Nothing, Tuple{Real,Real}}`: Optional bounds for rejection sampling on the intercept parameter (e.g., `(1.0, 10.0)`)

**Returns:** A matrix of posterior samples with shape `(n_params, n_samples)`.

**Example:**
```julia
K = 5
data = rand(1:50, 2^K - 1)

# Basic usage
samples = infer_npe(K, 0, 0, data; n_samples=5000)

# With prior bounds on intercept (recommended)
samples = infer_npe(K, 0, 0, data; n_samples=5000, intercept_bounds=(1.0, 10.0))

# Compute posterior summaries
using Statistics
posterior_means = mean(samples, dims=2)
posterior_stds = std(samples, dims=2)
```

### `prepare_data(data; censoring=false)`

Prepare raw count data for the neural estimators. This is called internally by `infer_nbe` and `infer_npe`, but can be used directly for custom workflows.

**Arguments:**
- `data::AbstractVector`: Vector of counts. Negative values indicate censored observations.
- `censoring::Bool`: If `true`, appends a censoring indicator vector (default: `false`)

**Returns:** A `Float32` matrix suitable for input to the neural estimators.

## Model Parameters

The estimators return parameters for a log-linear model:

```
log(λ_S) = α + Σᵢ βᵢ I(i ∈ S) + Σᵢ<ⱼ γᵢⱼ I(i ∈ S) I(j ∈ S)
```

Where:
- `α` (intercept): Base log-rate parameter
- `βᵢ` (main effects): Effect of list `i` on capture probability
- `γᵢⱼ` (interactions): Pairwise interaction between lists `i` and `j`

**Parameter ordering in output:**
1. Intercept (1 parameter)
2. Main effects β₁, β₂, ..., βₖ (K parameters)
3. Interactions γ₁₂, γ₁₃, ..., γ₍ₖ₋₁₎ₖ (K choose 2 parameters)

Total parameters: `1 + K + K(K-1)/2`

## Handling Censored Data

When some cell counts are censored (e.g., suppressed for privacy), indicate censored values with `-1` in your data vector:

```julia
# Data with censored values (indicated by -1)
data = [45, 23, -1, 8, 15, -1, 3, ...]  # -1 means censored

# Use appropriate censoring bounds that match your model
result = infer_nbe(K, 0, 10, data)  # censoring_upper=10 means counts ≤10 are censored
```

## Input Data Format

The data vector should contain counts for each combination of list memberships, ordered by the binary representation of list inclusion:

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

## Available Models

Pre-trained models are included for:
- **K**: 3, 4, 5, 6 lists
- **Censoring**: Various lower/upper bound combinations
- **Architecture**: 256-width, 3-hidden-layer networks

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
@Manual{,
    title = {{NeuralEstimators}: Likelihood-Free Parameter Estimation
      using Neural Networks},
    author = {Matthew Sainsbury-Dale},
    year = {2024},
    note = {R package version 0.1-2},
    url = {https://CRAN.R-project.org/package=NeuralEstimators},
    doi = {10.32614/CRAN.package.NeuralEstimators},
  }

@Article{,
    title = {Likelihood-Free Parameter Estimation with Neural {B}ayes
      Estimators},
    author = {Matthew Sainsbury-Dale and Andrew Zammit-Mangion and
      Raphael Huser},
    journal = {The American Statistician},
    year = {2024},
    volume = {78},
    pages = {1--14},
    doi = {10.1080/00031305.2023.2249522},
  }
```

## License

MIT License
