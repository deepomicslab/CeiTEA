module EigenDR

export Z_from_labels, Z_to_labels, order_col, construct_parent_X, map_to_child
export ilp, intercomp, candidates_from_pairs, dedup, candidates_from_β, select_best
export se

export eigendr

using Clustering: ClusteringResult
using LinearAlgebra: Diagonal

struct EigenDRResult <: ClusteringResult
    orig_X::AbstractMatrix{Float64}
    X::AbstractMatrix{Float64}
    D::AbstractMatrix{Float64}
    assignments::Vector{Int64}
    obj_val::Float64
end

include("Utils.jl")
using .Utils: Z_from_labels, Z_to_labels, order_col, construct_parent_X, map_to_child

include("StructuralEntropy.jl")
using .StructuralEntropy: se

include("Solver.jl")
using .Solver:
    ilp, intercomp, candidates_from_pairs, dedup, candidates_from_β, select_best

include("Simulation.jl")
include("Visualization.jl")


function _single_layer(
    X::AbstractMatrix{Float64},
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :full,
    topology = true,
)::EigenDRResult
    # Normalize the input matrix
    orig_X = copy(X)
    X = X ./ sum(X)
    D = Diagonal(sum(X, dims = 1)[:])

    # if vrange == :full
    #     vrange = 1:size(X, 2)
    # end
    lbs_se = map(βrange) do β
        # println(β)
        raw_lbs, raw_ses =
            candidates_from_β(β, X, D, topology = topology, vrange = vrange)
        select_best(raw_lbs, raw_ses, X, D, topology = topology)
    end
    lbs = getindex.(lbs_se, 1)
    ses = getindex.(lbs_se, 2)
    # ses = se.(lbs, Ref(X), Ref(D), topology = topology)
    uniq_lbs, uniq_ses = dedup(lbs, ses)
    if length(uniq_lbs) == 1
        return EigenDRResult(orig_X, X, D, Z_to_labels(uniq_lbs[1]), uniq_ses[1])
    end
    best_lbs, best_se = select_best(uniq_lbs, uniq_ses, X, D, topology = topology)
    labels = Z_to_labels(best_lbs)
    return EigenDRResult(orig_X, X, D, labels, best_se)
end

function eigendr(
    X::AbstractMatrix{Float64};
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :full,
    topology = true,
    hierarchy = false,
)::Union{EigenDRResult,Vector{EigenDRResult}}
    bottom_layer_clusters = _single_layer(X, βrange, vrange, topology)
    if !hierarchy
        return bottom_layer_clusters
    end
end

end # module EigenDR