using CoordinatePartitioning: euclid_embed, partition

function get_2d_partitions(edm, ratio, num, strategy)
    new_loc, evals = euclid_embed(edm; centered=true)
    num_par = Int(ex["ratio"] * ex["n"])
    par = partition(evals, num_par, ex["strategy"])
    return [new_loc[:, par] for par in partitions]
end
