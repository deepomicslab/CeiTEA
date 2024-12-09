module Utils

export Z_from_labels,
    Z_to_labels,
    order_col,
    construct_parent_X,
    map_to_child,
    norm_labels,
    TreeNode,
    build_tree,
    get_leaf_nodes,
    get_tree_height,
    unbalanced_tree_to_labels,
    tree_to_newick,
    sortperm_by_labels

using ..CeiTEA: CeiTEAResult

using Pipe: @pipe
using Clustering: randindex

mutable struct TreeNode
    index::Union{Nothing,Int64}
    label::Union{Nothing,String}
    children::Union{Nothing,Vector{TreeNode}}
end

function _build_tree!(
    node::TreeNode,
    lbs::AbstractMatrix{Int64},
    idx::AbstractVector{Int64},
    orig_idx::Union{Nothing,AbstractVector{Int64}},
    level::Int64;
    leaf_labels::Union{Nothing,AbstractVector{String}} = nothing,
)
    if level == 0
        node.children = map(idx) do i
            if isnothing(leaf_labels)
                TreeNode(orig_idx[i], "leaf-$(orig_idx[i])", nothing)
            else
                TreeNode(orig_idx[i], "$(leaf_labels[orig_idx[i]])", nothing)
            end
        end
    else
        nc = unique(lbs[idx, level])
        if length(nc) > 1
            node.children = map(nc) do l
                cidx = findall(lbs[idx, level] .== l)
                cnode = TreeNode(nothing, nothing, [])
                _build_tree!(
                    cnode,
                    lbs,
                    idx[cidx],
                    orig_idx,
                    level - 1,
                    leaf_labels = leaf_labels,
                )
                cnode
            end
        else
            _build_tree!(node, lbs, idx, orig_idx, level - 1, leaf_labels = leaf_labels)
        end
    end
end

function build_tree(
    normed_lbs::AbstractMatrix{Int64};
    leaf_labels::Union{Nothing,AbstractVector{String}} = nothing,
    orig_idx::Union{Nothing,AbstractVector{Int64}} = nothing,
)::TreeNode
    n_layer = size(normed_lbs, 2)
    root = TreeNode(nothing, nothing, [])
    idx = axes(normed_lbs, 1)
    if isnothing(orig_idx)
        orig_idx = copy(idx)
    end
    _build_tree!(root, normed_lbs, idx, orig_idx, n_layer, leaf_labels = leaf_labels)
    root
end

function get_leaf_nodes(node::TreeNode)::Vector{TreeNode}
    # If the node has no children, it's a leaf node
    if isnothing(node.children)
        return [node]
    end

    # Otherwise, recursively find leaf nodes in children
    leaf_nodes = []
    for child in node.children
        append!(leaf_nodes, get_leaf_nodes(child))
    end

    return leaf_nodes
end

function get_tree_height(node::TreeNode)::Int64
    if isnothing(node.children)
        return 0
    end

    max_height = 0
    for child in node.children
        max_height = max(max_height, get_tree_height(child))
    end

    return max_height + 1
end

function _assign_labels!(labels_mat, labels, root)
    if isnothing(root.children)
        labels_mat[root.index] = copy(labels[1:end-1])
        return
    end

    for (i, child) in enumerate(root.children)
        push!(labels, i)
        _assign_labels!(labels_mat, labels, child)
        pop!(labels)
    end
end

function unbalanced_tree_to_labels(root::TreeNode)::AbstractMatrix{Int64}
    height = get_tree_height(root)
    labels = fill([], length(get_leaf_nodes(root)))
    _assign_labels!(labels, [], root)
    n_not_full = findall(length.(labels) .< height - 1)
    while !isempty(n_not_full)
        foreach(i -> push!(labels[i], 0), n_not_full)
        n_not_full = findall(length.(labels) .< height - 1)
    end
    labels_mat = zeros(Int64, length(labels), height - 1)

    foreach(reverse(1:height-1)) do i
        layer_lbs_s = @pipe map(l -> join(map(string, l[1:i]), "."), labels)
        mapping = @pipe unique(layer_lbs_s) |> Dict(zip(_, 1:length(_)))
        @pipe filter(s -> endswith(s, "0"), keys(mapping)) |>
              foreach(x -> mapping[x] = 0, _)
        layer_lbs = map(x -> mapping[x], layer_lbs_s)
        labels_mat[:, height-1-i+1] = layer_lbs
    end
    for i in reverse(axes(labels_mat, 2))[2:end]
        zero_idx = findall(labels_mat[:, i] .== 0)
        labels_mat[zero_idx, i] .=
            labels_mat[zero_idx, i+1] .+ maximum(labels_mat[:, i]) .+ 1
    end
    for i in axes(labels_mat, 2)
        mapping = @pipe unique(labels_mat[:, i]) |> Dict(zip(_, 1:length(_)))
        labels_mat[:, i] .= map(x -> mapping[x], labels_mat[:, i])
    end
    labels_mat
end


function _tree_to_newick(node::TreeNode)::String
    if isnothing(node.children)
        return node.label
    else
        return "($(join(map(_tree_to_newick, node.children), ",")))"
    end
end

function tree_to_newick(node::TreeNode)::String
    return "$(_tree_to_newick(node));"
end

function _fill_Z!(Z::BitMatrix, l::AbstractVector{Int})
    for (idx, k) in enumerate(unique(l)), i in axes(l, 1)
        if l[i] == k
            Z[i, idx] = !Z[i, idx]
        end
    end
end

function Z_from_labels(l::AbstractVector{Int})::BitMatrix
    Z = falses(size(l, 1), size(unique(l), 1))
    _fill_Z!(Z, l)
    Z
end

function Z_to_labels(Z::BitMatrix)::Vector{Int}
    @pipe Z .* collect(1:size(Z, 2))' |> sum(_, dims = 2) |> vec
end

function order_col(Z::BitMatrix)::BitMatrix
    @pipe findfirst.(eachcol(Z)) |> sortperm |> Z[:, _]
end

function construct_parent_X(cls_res::CeiTEAResult)::AbstractMatrix{Float64}
    lbs = Z_from_labels(cls_res.assignments)
    pX = zeros(size(lbs, 2), size(lbs, 2))
    foreach(CartesianIndices(pX)) do idx
        # pX[idx] = cls_res.orig_X[lbs[:, idx[1]], lbs[:, idx[2]]] |> sum
        pX[idx] = cls_res.input.orig_A[lbs[:, idx[1]], lbs[:, idx[2]]] |> sum
    end
    pX
end

function map_to_child(
    parent_labels::AbstractVector{Int64},
    child_labels::AbstractVector{Int64},
)::Vector{Int64}
    converting_lbs = Z_from_labels(parent_labels)
    base_lbs = Z_from_labels(child_labels)
    converted = @pipe map(eachcol(converting_lbs)) do l
        @pipe base_lbs[:, l] |> reduce(.|, eachcol(_))
    end |> hcat(_...)
    Z_to_labels(converted)
end

function norm_labels(cls::AbstractVector{CeiTEAResult})::AbstractMatrix{Int64}
    lbs = [cls[1].assignments]
    for i in eachindex(cls)[2:end]
        push!(lbs, map_to_child(cls[i].assignments, lbs[i-1]))
    end
    lbs = hcat(lbs...)
    if size(lbs, 2) > 1 && randindex(lbs[:, end], lbs[:, end-1])[1] == 1
        return lbs[:, 1:end-1]
    end
    return lbs
end

function sortperm_by_labels(labels::AbstractMatrix{Int64})::AbstractVector{Int64}
    col_sort = reverse(axes(labels, 2))
    si = sortperm(eachrow(labels), by = row -> getindex(row, col_sort))
    si
end

end # module Utils