#=
Artifact management for pretrained NeuralMSE models.

Artifact naming: neuralmse_models_k{K}_{type}_c{censoring_lower}[_b{N}]

Models are split by (K, model_type, censoring_lower). If a group still exceeds
1.9 GB, it is further split into numbered batches (_b1, _b2, ...). Batched
entries in Artifacts.toml carry `censoring_upper_min` and `censoring_upper_max`
fields so the correct batch can be identified without downloading anything.

For development/training, set NEURALMSE_MODELS_PATH to skip artifact downloads.
=#

using DataFrames
using Pkg.Artifacts: artifact_meta, artifact_path, ensure_artifact_installed
using Pkg.TOML

const ARTIFACT_PREFIX = "neuralmse_models_k"
const K_RANGE         = 3:15
const MODEL_TYPES     = [:nbe, :npe]

# Per-request cache: (model_type, K, censoring_lower, censoring_upper) => local path
const _path_cache         = Dict{Tuple{Symbol,Int,Int,Int}, String}()
const _artifacts_available = Ref{Union{Bool,Nothing}}(nothing)

#=
Internal helpers
=#

_artifact_base(K, model_type, censoring_lower) =
    "$(ARTIFACT_PREFIX)$(K)_$(model_type)_c$(censoring_lower)"

function _check_artifacts_available()
    if _artifacts_available[] === nothing
        try
            toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
            if isfile(toml)
                content = read(toml, String)
                has_entry    = occursin(ARTIFACT_PREFIX, content)
                has_placeholder = occursin("0000000000000000000000000000000000000000", content)
                _artifacts_available[] = has_entry && !has_placeholder
            else
                _artifacts_available[] = false
            end
        catch
            _artifacts_available[] = false
        end
    end
    return _artifacts_available[]
end

"""Download (if needed) and return the local path for a named artifact."""
function _install_artifact(artifact_name::String, meta::Dict)
    hash = Base.SHA1(meta["git-tree-sha1"])
    ensure_artifact_installed(artifact_name, meta, joinpath(dirname(@__DIR__), "Artifacts.toml"))
    return artifact_path(hash)
end

"""
Return the local artifact path for the models matching (K, model_type, censoring_lower,
censoring_upper). Downloads the artifact on first call; subsequent calls are cached.
"""
function _get_artifact_path_for_model(
    K::Int, model_type::Symbol, censoring_lower::Int, censoring_upper::Int
)
    cache_key = (model_type, K, censoring_lower, censoring_upper)
    haskey(_path_cache, cache_key) && return _path_cache[cache_key]

    toml_path = joinpath(dirname(@__DIR__), "Artifacts.toml")
    base_name = _artifact_base(K, model_type, censoring_lower)

    # 1. Try single (un-batched) artifact first
    meta = artifact_meta(base_name, toml_path)
    if meta !== nothing
        path = _install_artifact(base_name, meta)
        _path_cache[cache_key] = path
        return path
    end

    # 2. Try batched artifacts; pick by censoring_upper range stored in TOML
    for b in 1:50
        batch_name = "$(base_name)_b$(b)"
        meta = artifact_meta(batch_name, toml_path)
        meta === nothing && break

        cu_min = get(meta, "censoring_upper_min", -1)
        cu_max = get(meta, "censoring_upper_max", typemax(Int))

        if cu_min <= censoring_upper <= cu_max
            path = _install_artifact(batch_name, meta)
            _path_cache[cache_key] = path
            return path
        end
    end

    error(
        "No artifact found for K=$K model_type=$model_type " *
        "censor=($censoring_lower,$censoring_upper).\n" *
        "Run scripts/package_artifacts.jl to regenerate Artifacts.toml."
    )
end

#=
Development / custom path
=#

"""
    get_models_path() -> String

Return the local models directory. Only works in development mode
(NEURALMSE_MODELS_PATH must be set). In artifact mode use the
`load_pretrained_model` / `find_pretrained_model` API directly.
"""
function get_models_path()
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    isempty(p) && error(
        "get_models_path() requires NEURALMSE_MODELS_PATH to be set.\n" *
        "Models are split into per-(K,type,censoring) artifacts in artifact mode.\n" *
        "Use load_pretrained_model(K=..., model_type=...) to download on demand."
    )
    isdir(p) || error("NEURALMSE_MODELS_PATH='$p' does not exist")
    return p
end

ensure_models_available() = get_models_path()

#=
Public API
=#

"""
    list_available_models() -> DataFrame

List all pretrained models. Requires NEURALMSE_MODELS_PATH (development mode).
"""
function list_available_models()
    return list_models(get_models_path())
end

"""
    list_available_models(K::Int, model_type::Symbol) -> DataFrame

List pretrained models for a specific (K, model_type). Downloads both censoring
artifacts for this group if not already present.
"""
function list_available_models(K::Int, model_type::Symbol)
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    if !isempty(p)
        df = list_models(p)
        return filter(row -> row.K == K && row.model_type == model_type, df)
    end
    _check_artifacts_available() || error(
        "Artifacts not available. Run scripts/package_artifacts.jl and upload to GitHub."
    )
    # Merge both censoring_lower groups for this (K, type)
    dfs = DataFrame[]
    toml_path = joinpath(dirname(@__DIR__), "Artifacts.toml")
    for cl in [0, 1]
        base = _artifact_base(K, model_type, cl)
        # collect all artifact names for this cl group (single or batched)
        names = String[]
        if artifact_meta(base, toml_path) !== nothing
            push!(names, base)
        else
            for b in 1:50
                batch = "$(base)_b$(b)"
                artifact_meta(batch, toml_path) === nothing && break
                push!(names, batch)
            end
        end
        for name in names
            meta = artifact_meta(name, toml_path)
            meta === nothing && continue
            path = _install_artifact(name, meta)
            push!(dfs, list_models(path))
        end
    end
    isempty(dfs) && error("No artifacts found for K=$K model_type=$model_type")
    return vcat(dfs...)
end

"""
    find_pretrained_model(; K, model_type, censoring_lower=0, censoring_upper=0)

Find a pretrained model ID. Downloads the relevant artifact if needed.
"""
function find_pretrained_model(;
    K::Int, model_type::Symbol,
    censoring_lower::Int=0, censoring_upper::Int=0
)
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    models_dir = isempty(p) ?
        _get_artifact_path_for_model(K, model_type, censoring_lower, censoring_upper) :
        (isdir(p) ? p : error("NEURALMSE_MODELS_PATH='$p' does not exist"))
    return find_model(models_dir; K=K, model_type=model_type,
                      censoring_lower=censoring_lower, censoring_upper=censoring_upper)
end

"""
    load_pretrained_model(; K, model_type, censoring_lower=0, censoring_upper=0)

Load a pretrained model by configuration. Downloads the relevant artifact if needed.

# Example
```julia
(point_est, ci_est), cfg = load_pretrained_model(K=5, model_type=:nbe)
estimator, cfg           = load_pretrained_model(K=5, model_type=:npe, censoring_upper=8)
```
"""
function load_pretrained_model(;
    K::Int, model_type::Symbol,
    censoring_lower::Int=0, censoring_upper::Int=0
)
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    models_dir = isempty(p) ?
        _get_artifact_path_for_model(K, model_type, censoring_lower, censoring_upper) :
        (isdir(p) ? p : error("NEURALMSE_MODELS_PATH='$p' does not exist"))
    return load_model(models_dir; K=K, model_type=model_type,
                      censoring_lower=censoring_lower, censoring_upper=censoring_upper)
end

"""
    load_pretrained_model(model_id, K, model_type, censoring_lower=0, censoring_upper=0)

Load by model ID. `K`, `model_type`, and censoring bounds are needed to select the artifact.
"""
function load_pretrained_model(
    model_id::Int, K::Int, model_type::Symbol,
    censoring_lower::Int=0, censoring_upper::Int=0
)
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    models_dir = isempty(p) ?
        _get_artifact_path_for_model(K, model_type, censoring_lower, censoring_upper) :
        (isdir(p) ? p : error("NEURALMSE_MODELS_PATH='$p' does not exist"))
    return load_model(models_dir, model_id)
end

"""
    pretrained_model_exists(; K, model_type, censoring_lower=0, censoring_upper=0) -> Bool

Check if a pretrained model exists. Downloads the relevant artifact if needed.
"""
function pretrained_model_exists(;
    K::Int, model_type::Symbol,
    censoring_lower::Int=0, censoring_upper::Int=0
)
    p = get(ENV, "NEURALMSE_MODELS_PATH", "")
    models_dir = isempty(p) ?
        _get_artifact_path_for_model(K, model_type, censoring_lower, censoring_upper) :
        (isdir(p) ? p : error("NEURALMSE_MODELS_PATH='$p' does not exist"))
    return model_exists(models_dir; K=K, model_type=model_type,
                        censoring_lower=censoring_lower, censoring_upper=censoring_upper)
end

"""
    print_available_models()

Print a summary of all pretrained models (development mode only).
"""
function print_available_models()
    df = list_available_models()
    println("Available NeuralMSE Pretrained Models")
    println("=====================================")
    for mtype in [:nbe, :npe]
        sub = filter(row -> row.model_type == mtype, df)
        nrow(sub) == 0 && continue
        type_name = mtype == :nbe ? "NBE" : "NPE"
        println("$type_name: $(nrow(sub)) models  K=$(minimum(sub.K))–$(maximum(sub.K))")
    end
end
