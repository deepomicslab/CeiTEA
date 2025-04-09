using CSV
using DataFrames
using CeiTEA
using CeiTEA.Utils

# Read the adjacency matrix from a CSV file
A = CSV.read("SBM_100_10_strong_noise0_adj.csv", DataFrame) |> Matrix

# Run the CeiTEA algorithm and get the plain clusters
res = ceitea(A, hierarchy = true, vrange = 1:10, topology = true)

# Build the tree bottom-up
norm_lbs = norm_labels(res)
tree = build_tree(norm_lbs, leaf_labels = ["n$(i)" for i in axes(A, 1)])

# Save the tree to a file in Newick format
open( "SBM_100_10_strong_noise0_ceitea_TE_tree.nwk", "w") do io
    write(io, tree_to_newick(tree))
end
