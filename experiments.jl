using OptiTest: run, plot
using CoordinatePartitioning: rand_edm, euclid_embed, partition, build_edms, construct
using JuMP: optimize!, solve_time, objective_value, objective_bound
using GLPK: GLPK

function rand_ball_edm(num::Integer, coords::Integer)
    return nothing
end

function generic_solve(experiment::AbstractDict)::AbstractDict
    # get problem information
    if experiment["type"] == "box"
        edm = rand_edm(experiment["n"], experiment["s"])
    elseif experiment["type"] == "ball"
        edm = rand_ball_edm(experiment["n"], experiment["s"])
    else
        throw(ArgumentError("Not a valid EDM type!"))
    end
    cardinality = experiment["p"]

    # construct partitions
    new_loc, evals = euclid_embed(edm; centered=true)
    par = partition(
        evals,
        Int(ceil(experiment["partition_ratio"] * experiment["n"])),
        experiment["strategy"],
    )
    edms = build_edms(new_loc, par)

    # construct model and solve
    mdl, num_cuts = construct(edms, cardinality, GLPK)
    optimize!(mdl)

    # get results
    experiment["resultant_partitions"] = length(par)
    experiment["solve_time"] = solve_time(mdl)
    experiment["obj_val"] = objective_value(mdl)
    experiment["best_bound"] = objective_bound(mdl)
    experiment["cuts"] = num_cuts[]

    return experiment
end

experiment = Dict(
    "type" => "box",
    "n" => 10,
    "s" => 2,
    "p" => 4,
    "partition_ratio" => 0.25,
    "strategy" => "random",
)
run(experiment, generic_solve)
