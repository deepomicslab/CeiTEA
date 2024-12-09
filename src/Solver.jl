module Solver

export ilp,
    intercomp, candidates_from_pairs, dedup, candidates_from_beta, select_best_lb

using Pipe: @pipe
using JuMP, HiGHS
using LinearAlgebra: Diagonal, eigen
using Clustering: dbscan, pairwise, randindex
using ThreadPools: bmap

using ..Utils: Z_from_labels, Z_to_labels
using ..Entropy: entropy
using ..CeiTEA: CEITEA_INPUT


function ilp(
    L::AbstractMatrix{Bool},
    metrics::AbstractVector{Float64};
    used::Union{BitVector,Nothing} = nothing,
    K::Union{Symbol,Int64} = :auto,
)
    model = Model(HiGHS.Optimizer)
    set_attribute(model, "output_flag", false)
    set_attribute(model, "log_to_console", false)
    set_attribute(model, "parallel", "on")
    # set_attribute(model, "threads", 64)
    @variable(model, x[1:size(L, 2)], Bin)
    @constraint(model, L * x .== 1)
    if !isnothing(used) && any(used)
        @constraint(model, x[hide] .== 0)
    end
    if K != :auto
        @constraint(model, sum(x) .== K)
    end
    @objective(model, Min, x' * metrics)
    optimize!(model)

    obj_val = objective_value(model)
    x_val = @pipe value.(x) |> round.(_) |> Bool.(_)
    obj_val, x_val
end

function intercomp(x::AbstractVector{Bool}, y::AbstractVector{Bool};)::BitMatrix
    inter = x .& y
    if !any(inter)
        return hcat([x, y]...)
    end
    x_rm_inter = x .⊻ inter
    y_rm_inter = y .⊻ inter
    ret = Set{AbstractVector{Bool}}()
    # push!(ret, [inter, x_rm_inter, y_rm_inter]...)
    if entropy(inter) + entropy(x_rm_inter) < entropy(x)
        push!(ret, [inter, x_rm_inter]...)
    else
        push!(ret, x)
    end
    if entropy(inter) + entropy(y_rm_inter) < entropy(y)
        push!(ret, [inter, y_rm_inter]...)
    else
        push!(ret, y)
    end
    return hcat(ret...)
end

@views function intercomp(L::BitMatrix)::BitMatrix
    @pipe map(axes(L, 2)) do i
        intercomp.(eachcol(L)[i+1:end], Ref(L[:, i]))
    end |> vcat(_...) |> hcat(_...)
end

function intercomp(L1::AbstractMatrix{Bool}, L2::AbstractMatrix{Bool})::BitMatrix
    @pipe [L1 L2] |> unique(eachcol(_)) |> hcat(_...) |> intercomp(_)
end

@views function intercomp(L::Vector{BitMatrix})::BitMatrix
    @pipe map(axes(L, 2)) do i
        intercomp.(L[i+1:end], Ref(L[i]))
    end |> vcat(_...) |> hcat(_...)
end

function candidates_from_pairs(
    l1::BitMatrix,
    l2::BitMatrix;
    K::Union{Symbol,Int64} = :auto,
)::Tuple{BitMatrix,Float64}
    # println(i1, i2)
    clbs = @pipe intercomp(l1, l2) |> unique(eachcol(_)) |> hcat(_...)
    cse = entropy.(eachcol(clbs))
    # idx = findall(cse .< 0)
    # candidate_lbs = clbs[:, idx]
    # candidate_se = cse[idx]
    obj_se, x = ilp(clbs, cse, K = K)
    clbs[:, x], obj_se
end

function dedup(
    lbs::AbstractVector{BitMatrix},
    ses::AbstractVector{Float64},
)::Tuple{Vector{BitMatrix},Vector{Float64}}
    ri = pairwise((x, y) -> randindex(x, y)[1], Z_to_labels.(lbs))
    rid = maximum(ri) .- ri
    res = dbscan(rid, 0, metric = nothing)
    c = Z_from_labels(res.assignments)
    uniq_lbs = getindex.(getindex.(Ref(lbs), eachcol(c)), 1)
    uniq_ses = getindex.(getindex.(Ref(ses), eachcol(c)), 1)
    uniq_lbs, uniq_ses
end

function candidates_from_β(
    β::Float64,
    vrange::Union{AbstractVector{Int64},Symbol} = :auto,
)::Tuple{Vector{BitMatrix},Vector{Float64}}
    print("\rComputing candidates from β = $β", " "^10)
    flush(stdout)
    _, v = eigen(β * CEITEA_INPUT.D - CEITEA_INPUT.A)
    lbs = @pipe Int64.(v .>= 0) |>
          unique(eachcol(_)) |>
          Z_from_labels.(_) |>
          filter(l -> size(l, 2) > 1, _)
    # ses = entropy.(lbs)
    ses = bmap(entropy, lbs)
    # neg_i = findall(ses .< 0)
    # lbs = lbs[neg_i]
    # ses = ses[neg_i]
    if vrange != :auto
        si = sortperm(ses)
        lbs = lbs[si[vrange]]
        ses = ses[si[vrange]]
    end

    if length(lbs) > 1
        lbs, ses = dedup(lbs, ses)
    end
    si = sortperm(ses)
    lbs[si], ses[si]
end

function iter_mlbs!(
    idx::BitVector,
    lbs::AbstractVector{BitMatrix},
    curr_lbs::BitMatrix;
    K::Union{Symbol,Int64} = :auto,
)
    # mlbs = select_for_pair.(lbs[idx], Ref(curr_lbs), Ref(X), Ref(D))
    mlbs = bmap(findall(idx)) do i
        candidates_from_pairs(lbs[i], curr_lbs, K = K)
    end
    lbs = getindex.(mlbs, 1)
    ses = getindex.(mlbs, 2)
    imin = argmin(ses)
    # println(findall(isapprox.(ses, ses[imin], atol = eps(Float32))))
    idx[findall(idx)[imin]] = false
    lbs[imin], ses[imin]
end

function select_best(
    lbs::AbstractVector{BitMatrix},
    ses::AbstractVector{Float64};
    K::Union{Symbol,Int64} = :auto,
)
    if length(lbs) == 1
        return lbs[1], ses[1]
    end
    idx = trues(length(lbs))
    idx[1] = false
    curr_lbs = lbs[1]
    prev_se = ses[1]
    curr_se = 0.0
    while true
        # println("prev_se: $prev_se")
        # flush(stdout)
        curr_lbs, curr_se = iter_mlbs!(idx, lbs, curr_lbs, K = K)
        if isapprox(curr_se, prev_se, atol = eps(Float32))
            break
        end
        if !any(idx)
            break
        end
        prev_se = curr_se
    end
    curr_lbs, curr_se
end

end # module Solver
