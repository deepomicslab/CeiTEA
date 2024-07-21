module EigenDR

export Z_from_labels, Z_to_labels, order_col, construct_parent_X, map_to_child
export ilp, intercomp, candidates_from_pairs, dedup, candidates_from_β, select_best
export entropy, set_entropy_params

export eigendr

using Clustering: ClusteringResult
using LinearAlgebra: Diagonal, diagind

mutable struct EigenDRInput
    orig_A::Union{AbstractMatrix{Float64},Nothing}
    A::Union{AbstractMatrix{Float64},Nothing}
    D::Union{Diagonal{Float64},Nothing}
end

const global EIGENDR_INPUT = EigenDRInput(nothing, nothing, nothing)

struct EigenDRResult <: ClusteringResult
    input::EigenDRInput
    assignments::Vector{Int64}
    obj_val::Float64
    # raw_candidates::Any
    # β_best_candidates::Any
    # best_candidates::BitMatrix
end

include("Utils.jl")
using .Utils: Z_from_labels, Z_to_labels, order_col, construct_parent_X, map_to_child

include("Entropy.jl")
using .Entropy: entropy, set_entropy_params

include("Solver.jl")
using .Solver:
    ilp, intercomp, candidates_from_pairs, dedup, candidates_from_β, select_best

include("Simulation.jl")
include("Visualization.jl")

function set_input(;
    orig_A::Union{AbstractMatrix{Float64},Nothing},
    A::Union{AbstractMatrix{Float64},Nothing},
    D::Union{Diagonal{Float64},Nothing},
)
    global EIGENDR_INPUT
    if !isnothing(orig_A)
        EIGENDR_INPUT.orig_A = orig_A
    end
    if !isnothing(A)
        EIGENDR_INPUT.A = A
    end
    if !isnothing(D)
        EIGENDR_INPUT.D = D
    end
end

function _single_layer(
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :full,
)::EigenDRResult
    # if vrange == :full
    #     vrange = 1:size(X, 2)
    # end
    all_raw_lbs = []
    lbs_se = map(βrange) do β
        # println(β)
        raw_lbs, raw_ses = candidates_from_β(β, vrange)
        push!(all_raw_lbs, raw_lbs)
        select_best(raw_lbs, raw_ses)
    end
    lbs = getindex.(lbs_se, 1)
    ses = getindex.(lbs_se, 2)
    # ses = se.(lbs, Ref(X), Ref(D), topology = topology)
    uniq_lbs, uniq_ses = dedup(lbs, ses)
    if length(uniq_lbs) == 1
        return EigenDRResult(
            EIGENDR_INPUT,
            Z_to_labels(uniq_lbs[1]),
            uniq_ses[1],
            # all_raw_lbs,
            # uniq_lbs,
            # uniq_lbs,
        )
    end
    best_lbs, best_se = select_best(uniq_lbs, uniq_ses)
    labels = Z_to_labels(best_lbs)
    return EigenDRResult(
        EIGENDR_INPUT,
        labels,
        best_se,
        # all_raw_lbs,
        # uniq_lbs,
        # best_lbs,
    )
end

function eigendr(
    X::AbstractMatrix{Float64};
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :full,
    topology = true,
    tol = eps(Float32),
    hierarchy = false,
)::Union{EigenDRResult,Vector{EigenDRResult}}
    A = X
    A[diagind(A)] .= 0
    A = A ./ sum(A)
    set_input(orig_A = X, A = A, D = Diagonal(sum(A, dims = 1)[:]))
    set_entropy_params(topology = topology, tol = tol)
    bottom_layer_clusters = _single_layer(βrange, vrange)
    if !hierarchy
        return bottom_layer_clusters
    end
end

end # module EigenDR