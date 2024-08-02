using LinearAlgebra: norm
using CoordinatePartitioning:
    build_edm, build_edms, euclid_embed, partition, STRATEGIES_ALL, isedm

using Plots

include("exclusion_zones.jl")

function force_2_cols(mat)
    if ndims(mat) != 2
        @warn "Matrix has $(ndims(mat)) dimension but should have 2!"
        return mat
    end
    cols = size(mat)[2]
    if cols > 2
        @warn "Removing a column from a matrix!"
        return mat[:, 1:2]
    end
    if cols == 2
        return mat
    end
    if cols == 1
        return hcat(mat, zero(mat[:, 1]))
    end
    # col=0
    return reshape([], 0, 2)
end

function R2_partitions(edm, strategy)
    new_loc, evals = euclid_embed(edm; centered=true)
    num_par = round(Int, ceil(size(new_loc)[2] / 2))
    partitions = partition(evals, num_par, strategy)
    new_locs = [force_2_cols(new_loc[:, par]) for par in partitions]
    return new_locs
end

function plot_partition(locations, cut=nothing)
    s = scatter(eachcol(force_2_cols(locations))...)
    if !isnothing(cut)
        plot_threshold!(locations[cut, :]; squared=true, other_locs=locations)
    end
    return s
end

function plot_partitions(locations, strategy="random", cut=nothing; kwargs...)
    edm = build_edm(locations)
    return plot(
        (plot_partition(loc, cut) for loc in R2_partitions(edm, strategy))...;
        link=:all,
        legend=false,
        ticks=false,
        kwargs...,
    )
end

function plot_orig_and_partitions(locations, strategy="random", cut=nothing; kwargs...)
    orig = scatter(eachcol(force_2_cols(locations))...; legend=false, aspect_ratio=:equal)
    plot_threshold!(locations[cut, :]; other_locs=locations)
    par = plot_partitions(locations, strategy, cut)
    return plot(orig, par; layout=(2, 1), kwargs...)
end
