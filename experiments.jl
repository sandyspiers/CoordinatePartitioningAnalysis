using Base: Generator
using OptiTester: OptiTest, TestRun, Iterable, FlattenIterable, Seed, run
using OptiTester: DataFrame, PerformanceProfile
using CoordinatePartitioning: rand_loc_cube, rand_loc_ball, build_edm
using CoordinatePartitioning: euclid_embed, partition, build_edms, construct
using CoordinatePartitioning: STRATEGIES
using Distributed: @everywhere
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
using Gurobi

@everywhere function solve(test::TestRun)::TestRun
    return test
    edm = build_edm(test.generator(test.n, test.s))
    # construct partitions
    if test.solver == :coordinate_partitioning
        t1 = time()
        new_loc, evals = euclid_embed(edm; centered=true)
        num_par = Int(test.ratio * test.n)
        par = partition(evals, num_par, test.strategy)
        edms = build_edms(new_loc, par)
        test.setup_time = time() - t1
        test.recovered_coords = size(new_loc)[2]
        test.resultant_partitions = length(par)
    else
        edms = edm
        test.setup_time = 0.0
        test.recovered_coords = 0
        test.resultant_partitions = 0
    end

    # construct model and solve
    if test.solver != :quadratic
        mdl, num_cuts = construct(edms, test.p, Gurobi)
    else
        mdl = JumpModel(Gurobi.Optimizer)
        # add variables and cardinality constraint
        @variable(mdl, 0 <= location_vars[1:(test.n)] <= 1, Bin)
        @constraint(mdl, sum(location_vars) == test.p)
        @objective(mdl, Max, x * edm * x')
        num_cuts = Ref(0)
    end
    set_time_limit_sec(mdl, test.time_limit)
    optimize!(mdl)

    # get results
    test.obj_val = objective_value(mdl)
    test.best_bound = objective_bound(mdl)
    test.gap = relative_gap(mdl)
    test.cuts = num_cuts[]
    test.solve_time = solve_time(mdl)
    test.total_time = test.setup_time + test.solve_time

    return test
end

cube = OptiTest(;#
    generator=Iterable(rand_loc_cube, rand_loc_ball),
    n=Iterable(10, 20),
    s=Iterable(2, 5),
    p=Iterable(0.1, 0.2),
    backend=Iterable(
        (
            solver=:coordinate_partitioning,
            ratio=Iterable(0.1, 0.25, 0.5, 0.75),
            strategy=Iterable(STRATEGIES),
        ),
        (solver=:coordinate_partitioning, strategy=:total),
        (solver=:cut_plane,),
        (solver=:quadratic,),
    ),
)
cube_results = run(cube, solve)
