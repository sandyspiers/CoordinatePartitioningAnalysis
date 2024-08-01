using LinearAlgebra: norm
using CoordinatePartitioning: build_edm

using Plots

rand_loc(n::Integer, coords::Integer=2) = rand(n, coords)
function rand_cir(n::Integer, coords::Integer=2)
    l = randn(n, coords)
    n = repeat(norm.(eachrow(l)); inner=(1, coords))
    return (l ./ n) ./ 2 .+ 0.5
end

function increased_range(collection, steps, increase)
    return increased_range(minimum(collection), maximum(collection), steps, increase)
end
function increased_range(min, max, steps, increase)
    rng = max - min
    min -= rng * increase
    max += rng * increase
    return range(min, max, steps)
end

function sum_distance(points; squared=false)
    function dist(loc)
        if squared
            return sum(norm(loc - pt)^2 for pt in eachrow(points))
        else
            return sum(norm(loc - pt) for pt in eachrow(points))
        end
    end
    return dist
end

function total_distance(points; squared=false)
    return sum(sum_distance(points; squared=squared)(pt) for pt in eachrow(points)) / 2
end

function offsets(cuts::Vector{Matrix{T}} where {T}, lb=0)
    fy = total_distance.(cuts)
    lb = max(lb, maximum(fy))
    p = size(first(cuts))[1]
    return (lb .- fy) ./ p
end

function sum_threshold(points::Matrix{T} where {T}, offset=0; squared=false)
    sd = sum_distance(points; squared=squared)
    thresholds = [sd(pt) + offset for pt in eachrow(points)]
    return loc -> sum(sd(loc) .< thresholds)
end

function sum_threshold(cuts::Vector{Matrix{T}} where {T}; use_lb=true, lb=0, squared=false)
    if use_lb || lb > 0
        st = (
            sum_threshold(cut, offset; squared=squared) for
            (cut, offset) in zip(cuts, offsets(cuts, lb))
        )
    else
        st = (sum_threshold(cut; squared=squared) for cut in cuts)
    end
    max_thresholds(loc) = maximum(s(loc) for s in st)
    return max_thresholds
end

plot_threshold!(cut::Matrix{T} where {T}; kwargs...) = plot_threshold!([cut]; kwargs...)
function plot_threshold!(
    cuts::Vector{Matrix{T}} where {T};
    use_lb=false,
    lb=0,
    steps=300,
    squared=false,
    other_locs=nothing,
)
    st = sum_threshold(cuts; use_lb=use_lb, lb=lb, squared=squared)

    locs = isnothing(other_locs) ? vcat(cuts...) : vcat(other_locs, cuts...)
    x, y = (increased_range(locs[:, c], steps, 0.2) for c in 1:2)
    z = ((x, y) -> st([x, y])).(x', y)

    CM = cgrad(:thermal; rev=true)
    heatmap!(x, y, z; aspect_ratio=:equal, alpha=0.6, colormap=CM, colorbar=false)
    for cut in cuts
        contri = invperm(sortperm([sum_distance(cut)(pt) for pt in eachrow(cut)]; rev=true))
        scatter!(
            cut[:, 1],
            cut[:, 2];
            zcolor=contri,
            colormap=CM,
            markersize=5,
            alpha=1,
            aspect_ratio=:equal,
            colorbar=false,
            legend=false,
        )
    end
    return scatter!()
end

plot_cut(cut; kwargs...) = plot_cuts(cut; kwargs...)
function plot_cuts(cuts; kwargs...)
    plot(; aspect_ratio=:equal)
    return plot_threshold!(cuts; kwargs...)
end

function plot_cuts_and_combined(cuts; squared=false)
    fy = plot(
        plot_cut.(cuts; use_lb=false, squared=squared)...;
        layout=(1, length(cuts)),
        aspect_ratio=:equal,
        link=:all,
    )
    all = plot(plot_cuts(cuts; use_lb=true, squared=squared); aspect_ratio=:equal)
    return plot(
        fy,
        all;
        layout=grid(2, 1; heights=[1, length(cuts)] ./ (1 + length(cuts))),
        link=:all,
    )
end

function plot_cuts_expanded_and_combined(cuts; squared=false)
    fy = plot(
        plot_cut.(cuts; use_lb=false, squared=squared)...;
        layout=(1, length(cuts)),
        aspect_ratio=:equal,
        link=:all,
    )
    LB = maximum(total_distance(cut; squared=squared) for cut in cuts)
    lb = plot(
        plot_cut.(cuts; lb=LB, squared=squared)...;
        layout=(1, length(cuts)),
        aspect_ratio=:equal,
        link=:all,
    )
    all = plot(plot_cuts(cuts; use_lb=true, squared=squared); aspect_ratio=:equal)
    return plot(
        fy,
        lb,
        all;
        layout=grid(3, 1; heights=[0.5, 0.5, length(cuts)] ./ (1 + length(cuts))),
        link=:all,
    )
end
