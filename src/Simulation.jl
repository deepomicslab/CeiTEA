module Simulation

export generate_similarity

using Random: rand, MersenneTwister

function generate_similarity(
    community_sizes,
    p_intra = 0.5,
    p_inter = 0.01;
    rand_weight = false,
    intra_weight = 10,
    inter_weight = 5,
)
    tn = sum(community_sizes)
    mat = zeros(tn, tn)
    start = 1
    for s in community_sizes
        intra = Int64.(rand(MersenneTwister(s), Float64, (s, s)) .< p_intra)
        if rand_weight
            intra =
                rand(MersenneTwister((s)^2), Float64, (s, s)) .* intra_weight .* intra
        end
        intra += intra'
        mat[start:(start+s-1), start:(start+s-1)] += intra
        start += s
    end
    inter = Int64.(rand(MersenneTwister(tn^2), Float64, (tn, tn)) .< p_inter)
    if rand_weight
        inter = rand(MersenneTwister(tn^2), Float64, (tn, tn)) .* inter_weight .* inter
    end
    inter += inter'
    mat += inter
    return mat
end

end # module Simulation