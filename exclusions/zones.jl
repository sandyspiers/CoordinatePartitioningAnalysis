using CoordinatePartitioning: build_edm
using Plots
# pythonplot()

using ProgressMeter
using LinearAlgebra: norm

rand_loc(n::Integer) = rand(n, 2)

function total_distance(points)
    return sum(sum_distance(points)(pt) for pt in eachrow(points)) / 2
end

function sum_distance(points)
    return loc -> sum(norm(loc .- point) for point in points)
end

function sum_threshold(points::Matrix{T} where {T}, offset=0)
    sd = sum_distance(points)
    return loc -> sum(sd(loc) < sd(pt) + offset for pt in eachrow(points))
end

function offsets(cuts::Vector{Matrix{T}} where {T})
    fy = total_distance.(cuts)
    lb = maximum(fy)
    p = size(first(cuts))[1]
    return (lb .- fy) ./ p
end

function sum_threshold(cuts::Vector{Matrix{T}} where {T}; use_lb=true)
    if use_lb
        st = (sum_threshold(cut, offset) for (cut, offset) in zip(cuts, offsets(cuts)))
    else
        st = (sum_threshold(cut) for cut in cuts)
    end
    max_thresholds(loc) = maximum(s(loc) for s in st)
    return max_thresholds
end

plot_threshold!(cut::Matrix{T} where {T}; kwargs...) = plot_threshold!([cut]; kwargs...)
function plot_threshold!(
    cuts::Vector{Matrix{T}} where {T}; use_lb=false, rng=range(0, 1, 500)
)
    st = sum_threshold(cuts; use_lb=use_lb)
    x = y = rng
    z = map(((x, y),) -> st([x, y]), Iterators.product(x, y))

    CM = cgrad(:thermal; rev=true)
    heatmap!(x, y, z; alpha=1, colormap=CM, colorbar=false)
    for cut in cuts
        contri = invperm(sortperm([sum_distance(cut)(pt) for pt in eachrow(cut)]; rev=true))
        scatter!(
            cut[:, 1],
            cut[:, 2];
            zcolor=contri,
            colormap=CM,
            markersize=5,
            alpha=1,
            colorbar=false,
        )
    end
end

p = 9
n = 45
m = 10

locs = rand_loc(n)
cuts = [rand_loc(p) for _ in 1:m]

single_fy = Plots.Plot[]
for cut in cuts
    plt = plot(; xlims=(0, 1), ylims=(0, 1))
    plot_threshold!(cut; use_lb=false)
    push!(single_fy, plt)
end
single_fy = plot(single_fy...; layout=(1, m))

all_lb = plot(; xlims=(0, 1), ylims=(0, 1))
plot_threshold!(cuts; use_lb=true)

plot(single_fy, all_lb; size=(1000, 2000), layout=grid(2, 1; heights=[0.3, 0.7]))
