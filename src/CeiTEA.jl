module CeiTEA

# export Z_from_labels,
#     Z_to_labels,
#     order_col,
#     construct_parent_X,
#     map_to_child,
#     TreeNode,
#     build_tree,
#     tree_to_newick,
#     norm_labels
export ilp,
    intercomp,
    candidates_from_pairs,
    dedup,
    candidates_from_β,
    select_best,
    local_diversify!
# export entropy, set_entropy_params

export ceitea

using Clustering: ClusteringResult, randindex
using LinearAlgebra: Diagonal, diagind
using Pipe: @pipe

mutable struct CeiTEAInput
    orig_A::Union{AbstractMatrix{Float64},Nothing}
    A::Union{AbstractMatrix{Float64},Nothing}
    D::Union{Diagonal{Float64},Nothing}
end

global CEITEA_INPUT = CeiTEAInput(nothing, nothing, nothing)

struct CeiTEAResult <: ClusteringResult
    input::CeiTEAInput
    assignments::Vector{Int64}
    obj_val::Float64
    # raw_candidates::Any
    # β_best_candidates::Any
    # best_candidates::BitMatrix
end

include("Utils.jl")
using .Utils:
    Z_from_labels,
    Z_to_labels,
    order_col,
    construct_parent_X,
    map_to_child,
    TreeNode,
    build_tree,
    tree_to_newick,
    norm_labels,
    unbalanced_tree_to_labels,
    get_leaf_nodes,
    get_tree_height

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
    global CEITEA_INPUT
    if !isnothing(orig_A)
        CEITEA_INPUT.orig_A = orig_A
    end
    if !isnothing(A)
        CEITEA_INPUT.A = A
    end
    if !isnothing(D)
        CEITEA_INPUT.D = D
    end
end

global all_raw_lbs = []
global all_raw_ses = []

function _single_layer(
    X::AbstractMatrix{Float64};
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :auto,
    topology = true,
    tol = eps(Float32),
    K::Union{Symbol,Int64} = :auto,
)::CeiTEAResult
    if vrange == :auto && size(X, 1) > 100
        vrange = 1:10
    end
    A = copy(X)
    A[diagind(A)] .= 0
    A = A ./ sum(A)
    set_input(orig_A = X, A = A, D = Diagonal(sum(A, dims = 1)[:]))
    set_entropy_params(topology = topology, tol = tol)
    global all_raw_lbs = []
    global all_raw_ses = []
    lbs_se = map(βrange) do β
        # println(β)
        raw_lbs, raw_ses = candidates_from_β(β, vrange)
        # delbs, deses = dedup(raw_lbs, raw_ses)
        # push!(all_raw_lbs, raw_lbs)
        # push!(all_raw_ses, raw_ses)
        select_best(raw_lbs, raw_ses)
    end
    println()
    flush(stdout)
    lbs = getindex.(lbs_se, 1)
    ses = getindex.(lbs_se, 2)
    # ses = se.(lbs, Ref(X), Ref(D), topology = topology)
    uniq_lbs, uniq_ses = dedup(lbs, ses)
    if length(uniq_lbs) == 1
        return CeiTEAResult(
            deepcopy(CEITEA_INPUT),
            Z_to_labels(uniq_lbs[1]),
            uniq_ses[1],
            # all_raw_lbs,
            # uniq_lbs,
            # uniq_lbs,
        )
    end
    best_lbs, best_se = select_best(uniq_lbs, uniq_ses, K = K)
    labels = Z_to_labels(best_lbs)
    return CeiTEAResult(
        deepcopy(CEITEA_INPUT),
        labels,
        best_se,
        # all_raw_lbs,
        # uniq_lbs,
        # best_lbs,
    )
end

function ceitea(
    X::AbstractMatrix{Float64};
    βrange::AbstractVector{Float64} = 0.01:0.01:2,
    vrange::Union{Symbol,AbstractVector{Int64}} = :auto,
    topology = true,
    tol = eps(Float32),
    hierarchy = false,
    K::Union{Symbol,Int64} = :auto,
)::Union{CeiTEAResult,Vector{CeiTEAResult}}
    # if size(X, 1) > 100
    #     vrange = 1:10
    # end
    layer_0 = _single_layer(
        X,
        βrange = βrange,
        vrange = vrange,
        topology = topology,
        tol = tol,
        K = K,
    )
    if !hierarchy
        return layer_0
    end

    # Hierarchical
    curr_layer = layer_0
    layers = CeiTEAResult[]
    while true
        push!(layers, curr_layer)
        next_layer_mat = construct_parent_X(curr_layer)
        if size(next_layer_mat, 1) <= 2
            break
        end
        next_layer = _single_layer(
            next_layer_mat,
            βrange = βrange,
            vrange = :auto,
            topology = topology,
            tol = tol,
            K = K,
        )
        if length(next_layer.assignments) == length(curr_layer.assignments) &&
           randindex(next_layer.assignments, curr_layer.assignments)[1] == 1
            break
        end
        curr_layer = next_layer
    end
    return layers
end

function _diversify_node!(
    node::TreeNode,
    queue::AbstractVector{TreeNode},
    X::AbstractMatrix{Float64},
)::TreeNode
    leafs = get_leaf_nodes(node)
    if length(leafs) == 1
        return node
    end
    leaf_idx = map(x -> x.index, leafs)
    leaf_mat = X[leaf_idx, leaf_idx]
    if all(isapprox.(leaf_mat, 0))
        return node
    end
    vrange = :auto
    if size(leaf_mat, 1) > 100
        vrange = 1:10
    end
    leaf_edr = ceitea(leaf_mat, hierarchy = true, vrange = vrange)
    if leaf_edr[1].obj_val < 0
        leaf_norm_labels = @pipe norm_labels(leaf_edr)
        subtree = build_tree(leaf_norm_labels, orig_idx = leaf_idx)
        node = subtree
        push!(queue, node)
    end
    return node
end

function local_diversify!(tree::TreeNode, X::AbstractMatrix{Float64})
    queue = [tree]
    while !isempty(queue)
        println("Pruning: $(length(queue)) left")
        flush(stdout)
        node = popfirst!(queue)
        node.children = map(x -> _diversify_node!(x, queue, X), node.children)
    end
end

end # module CeiTEA