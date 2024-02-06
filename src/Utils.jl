module Utils

using Random
using CairoMakie, Colors
using Graphs, SimpleWeightedGraphs, GraphMakie

# Write documentation for the following function
"""
    plot_adjacency(A::AbstractMatrix, labels=nothing; hmcolor=cgrad([:white, :red]))

Plot the adjacency matrix as a heatmap of a graph, with optional labels.

# Arguments
- `A::AbstractMatrix`: The adjacency matrix of the graph.
- `labels::Union{Nothing, Vector{Int}}`: The labels of the nodes.
- `hmcolor::ColorGradient`: The color gradient for the heatmap.

# Example
```julia
using Random
using Graphs, SimpleWeightedGraphs
using CairoMakie, Colors
using GraphMakie
using Utils

Random.seed!(0)
n = 10
g = SimpleWeightedGraph(n)
for i in 1:n
    for j in i+1:n
        if rand() < 0.2
            add_edge!(g, i, j, rand())
        end
    end
end
A = adjacency_matrix(g)
labels = rand(1:3, n)
plot_adjacency(A, labels)
```
"""
function plot_adjacency(A::AbstractMatrix, labels=nothing; hmcolor=cgrad([:white, :red]))
    mat_plot = log.(A .+ 1)
    mat_plot[diagind(mat_plot)] .= 0

    fig = Figure(size=(800, 800))

    if !isnothing(labels)
        colors = distinguishable_colors(size(unique(labels), 1), [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed=true)
        new_labels = copy(labels)
        if 0 in new_labels
            colors[1] = RGB(0, 0, 0)
            new_labels .+= 1
        end
        axtop = Axis(fig[1, 2])
        axleft = Axis(fig[2, 1])
        axhm = Axis(fig[2, 2], aspect=DataAspect())

        rowsize!(fig.layout, 2, Relative(0.9))
        colsize!(fig.layout, 2, Relative(0.9))

        hidespines!(axtop)
        hidedecorations!(axtop)
        hidespines!(axleft)
        hidedecorations!(axleft)
        hidedecorations!(axhm)

        xlims!(axtop, low=1, high=size(mat_plot, 1))
        ylims!(axtop, low=0, high=1)
        xlims!(axleft, low=0, high=1)
        ylims!(axleft, low=1, high=size(mat_plot, 1))
        barplot!(axtop, 1:size(mat_plot, 1), ones(size(mat_plot, 1)),
            color=colors[new_labels], gap=0, dodge_gap=0, width=1)
        barplot!(axleft, 1:size(mat_plot, 1), ones(size(mat_plot, 1)),
            color=colors[new_labels], gap=0, dodge_gap=0, width=1,
            direction=:x)
        heatmap!(axhm, mat_plot, colormap=hmcolor)
        axleft.yreversed = true
        axhm.yreversed = true
    else
        axhm = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
        heatmap!(axhm, mat_plot, colormap=hmcolor)
    end
    fig
end

"""
    generate_graph(n, k, p_intra_edge=0.5, p_inter_edge=0.01; rand_weight=false, seed=nothing)

Generate a random graph with `n` nodes and `k` communities.

# Arguments
- `n::Int`: The number of nodes in the graph.
- `k::Int`: The number of communities in the graph.
- `p_intra_edge::Float`: The probability of an edge within a community.
- `p_inter_edge::Float`: The probability of an edge between communities.
- `rand_weight::Bool`: Whether to generate random weights for the edges.
- `seed::Union{Nothing, Int}`: The seed for the random number generator.

# Example
```julia
using Random
using Graphs, SimpleWeightedGraphs
using Utils

n = 10
k = 3
g = generate_graph(n, k, p_intra_edge=0.5, p_inter_edge=0.01, rand_weight=true, seed=0)
```
"""
function generate_graph(n, k, p_intra_edge=0.5, p_inter_edge=0.01; rand_weight=false, seed=nothing)
    if !isnothing(seed)
        _state = copy(Random.default_rng())
    end
    g = SimpleWeightedGraph(n)
    community_sizes = rand(10:50, k)
    start = 1
    for s in community_sizes
        for i in start:(start+s-1), j in (i+1):(start+s-1)
            if rand() < p_intra_edge
                add_edge!(g, i, j, rand_weight ? rand() : 1)
            end
        end
    end
    for i in 1:n, j in (i+1):n
        if rand() < p_inter_edge
            add_edge!(g, i, j, rand_weight ? rand() : 1)
        end
    end

    if !isnothing(seed)
        Random.seed!(_state)
    end
    g
end

export plot_adjacency, generate_graph

end # module Utils