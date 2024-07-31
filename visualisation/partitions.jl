using LinearAlgebra: norm
using CoordinatePartitioning: build_edm, euclid_embed, partition, STRATEGIES_ALL

using Plots

include("exclusion_zones.jl")

function get_2d_partitions(edm, strategy)
    new_loc, evals = euclid_embed(edm; centered=true)
    num_par = round(Int, ceil(size(new_loc)[2] / 2))
    partitions = partition(evals, num_par, strategy)
    return [new_loc[:, par] for par in partitions]
end

function scatter_loc(locations, cut=nothing)
    coords = size(locations)[2]
    if coords == 0
        return scatter()
    end
    if coords == 1
        locations = hcat(locations, zero(locations[:, 1]))
    end
    s = scatter(locations[:, 1], locations[:, 2])
    if !isnothing(cut)
        plot_threshold!(locations[cut, :]; squared=true, other_locs=locations)
    end
    return s
end

function plot_partitions(locations, strategy="random", cut=nothing)
    edm = build_edm(locations)
    return plot(
        [scatter_loc(loc, cut) for loc in get_2d_partitions(edm, strategy)]...;
        link=:all,
        legend=false,
        ticks=false,
    )
end
