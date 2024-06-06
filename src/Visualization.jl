module Visualization

export plot_adjacency, plot_labels

using CairoMakie, Colors
using Pipe: @pipe
using LinearAlgebra: diagind

##
function plot_adjacency(
    A::AbstractMatrix,
    labels = nothing;
    hmcolor = cgrad([:white, :red]),
)
    mat_plot = log.(A .+ 1)
    mat_plot[diagind(mat_plot)] .= 0

    fig = Figure(; size = (800, 800))

    if !isnothing(labels)
        colors = distinguishable_colors(
            size(unique(labels), 1),
            [RGB(1, 1, 1), RGB(0, 0, 0)];
            dropseed = true,
        )
        new_labels = copy(labels)
        if 0 in new_labels
            colors[1] = RGB(0, 0, 0)
            new_labels .+= 1
        end
        axtop = Axis(fig[1, 2])
        axleft = Axis(fig[2, 1])
        axhm = Axis(fig[2, 2]; aspect = DataAspect())

        rowsize!(fig.layout, 2, Relative(0.9))
        colsize!(fig.layout, 2, Relative(0.9))

        hidespines!(axtop)
        hidedecorations!(axtop)
        hidespines!(axleft)
        hidedecorations!(axleft)
        hidedecorations!(axhm)

        heatmap!(axtop, labels, colormap = colors)
        heatmap!(axleft, labels', colormap = colors)
        heatmap!(axhm, mat_plot; colormap = hmcolor)
        axleft.yreversed = true
        axhm.yreversed = true
    else
        axhm = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true)
        heatmap!(axhm, mat_plot; colormap = hmcolor)
    end
    return fig
end

function plot_labels(labels::AbstractMatrix{Int64}; rvals = nothing, truth = nothing)
    colors = distinguishable_colors(
        size(unique(truth), 1),
        [RGB(1, 1, 1), RGB(0, 0, 0)];
        dropseed = true,
    )
    fig = Figure(size = (1000, 800))
    label_axis = 1
    if !isnothing(truth)
        axtop_truth = Axis(fig[1, 1])
        hidespines!(axtop_truth)
        hidedecorations!(axtop_truth)
        label_axis = 2
    end
    axdown_label = Axis(fig[label_axis, 1])
    if !isnothing(truth)
        rowsize!(fig.layout, 1, Relative(0.1))
        heatmap!(axtop_truth, truth', colormap = colors)
    end
    hidespines!(axdown_label)
    hidexdecorations!(axdown_label)
    if !isnothing(rvals)
        axright_vals = Axis(fig[2, 2])
        colsize!(fig.layout, 2, Relative(0.1))
        hidespines!(axright_vals)
        hideydecorations!(axright_vals)
        ylims!(axright_vals, low = 1 - 0.5, high = length(rvals) + 0.5)
        lines!(axright_vals, rvals, 1:length(rvals))
    end
    heatmap!(axdown_label, labels, colormap = [:yellow, :red])
    # axright.xreversed = true;
    fig
end

function plot_labels(labels::BitMatrix; rvals = nothing, truth = nothing)
    fig = Figure(size = (1000, 800))
    label_axis = 1
    if !isnothing(truth)
        axtop_truth = Axis(fig[1, 1])
        hidespines!(axtop_truth)
        hidedecorations!(axtop_truth)
        label_axis = 2
        colors = distinguishable_colors(
            size(unique(truth), 1),
            [RGB(1, 1, 1), RGB(0, 0, 0)];
            dropseed = true,
        )
    end
    axdown_label = Axis(fig[label_axis, 1])
    if !isnothing(truth)
        rowsize!(fig.layout, 1, Relative(0.1))
        heatmap!(axtop_truth, truth', colormap = colors)
    end
    hidespines!(axdown_label)
    hidexdecorations!(axdown_label)
    if !isnothing(rvals)
        rvals = Vector{Float64}(rvals)
        axright_vals = Axis(fig[label_axis, 2])
        colsize!(fig.layout, 2, Relative(0.1))
        hidespines!(axright_vals)
        hideydecorations!(axright_vals)
        ylims!(axright_vals, low = 1 - 0.5, high = length(rvals) + 0.5)
        lines!(axright_vals, rvals, 1:length(rvals))
        # axright_vals.xreversed = true
    end
    heatmap!(axdown_label, labels, colormap = [:white, :red])
    fig
end

##
end # module Visualization