using OptiTest: run, plot
using CoordinatePartitioning: rand_edm, euclid_embed, partition, build_edms, construct
using JuMP: Model as JumpModel
using JuMP:
    optimize!,
    solve_time,
    objective_value,
    objective_bound,
    relative_gap,
    set_time_limit_sec,
    @variable,
    @constraint,
    @objective
using SCIP: SCIP
using GLPK: GLPK

function rand_box_edm(num::Integer, coords::Integer)
    return rand_edm(num, coords)
end

function rand_ball_edm(num::Integer, coords::Integer)
    return nothing
end

function generic_solve(experiment::AbstractDict)::AbstractDict
    # shorthanding
    ex = experiment

    # get problem information
    if ex["type"] == "box"
        edm = rand_box_edm(ex["num"], ex["coords"])
    elseif ex["type"] == "ball"
        edm = rand_ball_edm(ex["num"], ex["coords"])
    else
        throw(ArgumentError("Not a valid EDM type!"))
    end
    cardinality = ex["card"]

    # construct partitions
    if ex["solver"] == "coordinate_partitioning"
        ex["setup_time"] = time()
        new_loc, evals = euclid_embed(edm; centered=true)
        num_par = Int(ex["ratio"] * ex["n"])
        par = partition(evals, num_par, ex["strategy"])
        edms = build_edms(new_loc, par)
        ex["setup_time"] = time() - ex["setup_time"]
        ex["recovered_coords"] = size(new_loc)[2]
        ex["resultant_partitions"] = length(par)
    else
        edms = edm
        ex["setup_time"] = 0.0
        ex["recovered_coords"] = 0
        ex["resultant_partitions"] = 0
    end

    # construct model and solve
    if ex["solver"] != "quadratic"
        mdl, num_cuts = construct(edms, cardinality, GLPK)
    else
        mdl = JumpModel(SCIP.Optimizer)
        # add variables and cardinality constraint
        @variable(mdl, 0 <= location_vars[1:ex["num"]] <= 1, Bin)
        @constraint(mdl, sum(location_vars) == ex["cardinality"])
        @objective(mdl, Max, x * edm * x')
        num_cuts = Ref(0)
    end
    set_time_limit_sec(mdl, ex["time_limit"])
    optimize!(mdl)

    # get results
    ex["obj_val"] = objective_value(mdl)
    ex["best_bound"] = objective_bound(mdl)
    ex["gap"]relative_gap(mdl)
    ex["cuts"] = num_cuts[]
    ex["solve_time"] = solve_time(mdl)
    ex["total_time"] = ex["setup_time"] + ex["solve_time"]

    return ex
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
