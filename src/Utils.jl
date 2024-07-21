module Utils

export Z_from_labels, Z_to_labels, order_col, construct_parent_X, map_to_child

using ..EigenDR: EigenDRResult

using Pipe: @pipe

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

function construct_parent_X(cls_res::EigenDRResult)::AbstractMatrix{Float64}
    lbs = Z_from_labels(cls_res.assignments)
    pX = zeros(size(lbs, 2), size(lbs, 2))
    foreach(CartesianIndices(pX)) do idx
        # pX[idx] = cls_res.orig_X[lbs[:, idx[1]], lbs[:, idx[2]]] |> sum
        pX[idx] = cls_res.input.A[lbs[:, idx[1]], lbs[:, idx[2]]] |> sum
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

end # module Utils