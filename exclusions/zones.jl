using CoordinatePartitioning: build_edm
using LinearAlgebra: norm
using Plots: plot, plot!, contour, contour!, scatter!

rand_loc(n::Integer) = rand(n, 2)

function agg_distance(location, points)
    return sum(norm(location .- point) for point in points)
end

function threshold(point, points)
    return location -> agg_distance(location, points) < agg_distance(point, points)
end

function plot_locs(locations)
    p = plot(; size=(2000, 2000), xlims=(0, 1), ylims=(0, 1))
    scatter!(locations[:, 1], locations[:, 2])
    return p
end

function plot_cut!(cut)
    scatter!(cut[:, 1], cut[:, 2]; color=:red)
    x = y = 0:0.001:1
    for point in eachrow(cut)
        contour!(x, y, (x, y) -> threshold(point, cut)([x, y]); levels=1, color=:red)
    end
end

loc = rand_loc(1)
cuts = (rand_loc(5) for _ in 1:2)
p = plot_locs(loc)
plot_cut!.(cuts)
p
