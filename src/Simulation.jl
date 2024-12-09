module Simulation

export generate_single_layer, generate_hierarchical_weights

using Random: rand, MersenneTwister, randperm
using LinearAlgebra: diagind
using Pipe: @pipe
using ..Utils:
    Z_from_labels,
    Z_to_labels,
    map_to_child,
    TreeNode,
    build_tree,
    unbalanced_tree_to_labels
using PyCall

function generate_single_layer(
    N::Int64,
    nc::Int64;
    p_intra::AbstractVector{Int64} = 80:100,
    p_inter::AbstractVector{Int64} = 1:5,
    edge_noise_level::Float64 = 0.1,
    seed::Int64 = 0,
)::Tuple{Vector{Int64},Matrix{Float64}}
    nx = pyimport("networkx")
    r = rand(MersenneTwister(nc^2 + seed), nc)
    sizes = @pipe r ./ sum(r) * N |> round.(Int64, _)
    p_mat = @pipe rand(MersenneTwister(nc^3 + seed), p_inter, nc, nc) ./ 100 |>
          _ + _' |>
          _ ./ 2
    d = rand(MersenneTwister(nc^4 + seed), p_intra, nc) ./ 100
    p_mat[diagind(p_mat)] .= d
    g = nx.stochastic_block_model(sizes, p_mat, seed = N * nc * seed)
    w = @pipe rand(MersenneTwister(nc^5 + seed), Float64, (sum(sizes), sum(sizes))) |>
          _ + _' |>
          _ ./ 2 * 10
    labels = @pipe g.nodes.data("block") |>
          Dict |>
          collect(values(_))[sortperm(collect(keys(_)))]
    aff = nx.adjacency_matrix(g).todense()
    edge_noise_idx =
        @pipe rand(MersenneTwister(nc^6 + seed), Float64, (sum(sizes), sum(sizes))) |>
              _ + _' |>
              _ ./ 2 |>
              findall(_ .< edge_noise_level)
    aff[edge_noise_idx] .= 1 .- aff[edge_noise_idx]
    used_mat = aff .* w
    labels, used_mat
end

function generate_hierarchical_weights(
    total_nodes::Int64,
    ref_ncs::AbstractVector{Int64}; # not guarantee, only for guarantee
    p_intra::AbstractVector{Int64} = 80:100,
    p_inter::AbstractVector{Int64} = 1:5,
    factor::Int64 = 10,
    edge_noise_level::Float64 = 0.1,
    seed::Int64 = 0,
)
    nx = pyimport("networkx")
    n_layer = length(ref_ncs)
    prev_N = total_nodes
    prev_sizes = ones(Int64, prev_N)
    A = zeros(prev_N, prev_N)
    ls = []
    for i = 1:n_layer
        N = prev_N
        nc = ref_ncs[i]
        l = @pipe [
                  randperm(MersenneTwister(nc + seed), nc)
                  rand(MersenneTwister(nc + 1 + seed), 1:nc, N - nc)
              ] |>
              sort |>
              Z_from_labels
        push!(ls, l)
        sizes = map(c -> sum(prev_sizes[c]), eachcol(l))
        p_mat = @pipe rand(MersenneTwister(nc^3 + seed), p_inter, nc, nc) ./ 100 |>
              _ + _' |>
              _ ./ 2
        d = rand(MersenneTwister(nc^4 + seed), p_intra, nc) ./ 100
        p_mat[diagind(p_mat)] .= d
        g = nx.stochastic_block_model(sizes, p_mat, seed = seed)
        # labels = @pipe g.nodes.data("block") |>
        #       Dict |>
        #       collect(values(_))[sortperm(collect(keys(_)))]
        # labels_mat = Z_from_labels(labels)
        aff = nx.adjacency_matrix(g).todense()
        w = @pipe rand(
                  MersenneTwister(nc^5 + seed),
                  Float64,
                  (sum(sizes), sum(sizes)),
              ) |>
              _ + _' |>
              _ ./ 2
        prev_j = 0
        for j in sizes
            w[prev_j+1:j+prev_j, prev_j+1:j+prev_j] .=
                w[prev_j+1:j+prev_j, prev_j+1:j+prev_j] .* factor^(n_layer - 1) /
                factor^(i - 1)
            prev_j += j
        end
        # w = w .* factor^(n_layer - 1) / factor^(i - 1)
        edge_noise_idx = @pipe rand(
                  MersenneTwister(nc^6 + seed),
                  Float64,
                  (sum(sizes), sum(sizes)),
              ) |>
              _ + _' |>
              _ ./ 2 |>
              findall(_ .< edge_noise_level)
        aff[edge_noise_idx] .= 1 .- aff[edge_noise_idx]
        adj = aff .* w
        A += adj

        prev_N = nc
        prev_sizes = sizes
    end
    labels = [Z_to_labels(l) for l in ls]
    norm_lbs = [labels[1]]
    for i in eachindex(labels)[2:end]
        push!(norm_lbs, map_to_child(labels[i], norm_lbs[i-1]))
    end
    norm_lbs = hcat(norm_lbs...)
    A, unbalanced_tree_to_labels(build_tree(norm_lbs)), norm_lbs
end

end # module Simulation
