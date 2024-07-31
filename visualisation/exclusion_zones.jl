using LinearAlgebra: norm
using CoordinatePartitioning: build_edm

using Plots
pythonplot()

rand_loc(n::Integer) = rand(n, 2)
function rand_cir(n::Integer)
    l = randn(n, 2)
    n = repeat(norm.(eachrow(l)); inner=(1, 2))
    return (l ./ n) ./ 2 .+ 0.5
end

function sum_distance(points)
    dist(loc) = sum(norm(loc - pt) for pt in eachrow(points))
    return dist
end

function total_distance(points)
    return sum(sum_distance(points)(pt) for pt in eachrow(points)) / 2
end

function offsets(cuts::Vector{Matrix{T}} where {T}, lb=0)
    fy = total_distance.(cuts)
    lb = max(lb, maximum(fy))
    p = size(first(cuts))[1]
    return (lb .- fy) ./ p
end

function sum_threshold(points::Matrix{T} where {T}, offset=0)
    sd = sum_distance(points)
    thresholds = [sd(pt) + offset for pt in eachrow(points)]
    return loc -> sum(sd(loc) .< thresholds)
end

function sum_threshold(cuts::Vector{Matrix{T}} where {T}; use_lb=true, lb=0)
    if use_lb || lb > 0
        st = (sum_threshold(cut, offset) for (cut, offset) in zip(cuts, offsets(cuts, lb)))
    else
        st = (sum_threshold(cut) for cut in cuts)
    end
    max_thresholds(loc) = maximum(s(loc) for s in st)
    return max_thresholds
end

plot_threshold!(cut::Matrix{T} where {T}; kwargs...) = plot_threshold!([cut]; kwargs...)
function plot_threshold!(
    cuts::Vector{Matrix{T}} where {T}; use_lb=false, lb=0, rng=range(0, 1, 500)
)
    st = sum_threshold(cuts; use_lb=use_lb, lb=lb)
    x = y = rng
    z = ((x, y) -> st([x, y])).(x', y)

    CM = cgrad(:thermal; rev=true)
    heatmap!(
        x,
        y,
        z;
        xlims=(0, 1),
        ylims=(0, 1),
        aspect_ratio=:equal,
        alpha=0.6,
        colormap=CM,
        colorbar=false,
    )
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
end

function plot_single_cut(p)
    cut = rand_cir(p)
    p = plot(; size=(1000, 1000))
    plot_threshold!(cut; use_lb=false)
    return p
end

function plot_multi_cut(p, m)
    cuts = [rand_loc(p) for _ in 1:m]
    p = plot(; xlims=(0, 1), ylims=(0, 1))
    plot_threshold!(cuts; use_lb=true)
    return p
end

function plot_single_cuts_and_contribution(p, n, m)
    locs = rand_loc(n)
    cuts = [rand_loc(p) for _ in 1:m]

    single_fy = Plots.Plot[]
    for cut in cuts
        plt = plot()
        plot_threshold!(cut; use_lb=false)
        push!(single_fy, plt)
    end
    single_fy = plot(single_fy...; layout=(1, m))

    all_lb = plot(; xlims=(0, 1), ylims=(0, 1))
    plot_threshold!(cuts; use_lb=true)
    scatter!(locs[:, 1], locs[:, 2]; markersize=5, alpha=1, legend=false)

    return plot(
        single_fy, all_lb; size=(1000, 1250), layout=grid(2, 1; heights=[0.25, 0.75])
    )
end

function plot_single_cuts_expanded_and_contribution(p, n, m)
    locs = rand_loc(n)
    cuts = [rand_loc(p) for _ in 1:m]

    single_fy = Plots.Plot[]
    for cut in cuts
        plt = plot()
        plot_threshold!(cut; use_lb=false)
        push!(single_fy, plt)
    end
    single_fy = plot(single_fy...; layout=(1, m))

    single_lb = Plots.Plot[]
    lb = maximum(total_distance(cut) for cut in cuts)
    for cut in cuts
        plt = plot()
        plot_threshold!(cut; lb=lb)
        push!(single_lb, plt)
    end
    single_lb = plot(single_lb...; layout=(1, m))

    all_lb = plot(; xlims=(0, 1), ylims=(0, 1))
    plot_threshold!(cuts; use_lb=true)
    scatter!(locs[:, 1], locs[:, 2]; markersize=5, alpha=1, legend=false)

    return plot(
        single_fy,
        single_lb,
        all_lb;
        size=(1000, 1500),
        layout=grid(3, 1; heights=[0.25, 0.25, 0.5]),
    )
end
