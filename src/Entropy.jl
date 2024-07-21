module Entropy

export set_entropy_params, entropy

using ..Utils: Z_from_labels
using ..EigenDR: EIGENDR_INPUT

using LinearAlgebra: Diagonal

mutable struct EntropyParams
    topology::Bool
    tol::Float64
end

const global ENTROPY_PARAMS = EntropyParams(true, eps(Float32))

function set_entropy_params(;
    topology::Union{Bool,Nothing} = nothing,
    tol::Union{AbstractFloat,Nothing} = nothing,
)
    global ENTROPY_PARAMS
    if !isnothing(topology)
        ENTROPY_PARAMS.topology = topology
    end
    if !isnothing(tol)
        ENTROPY_PARAMS.tol = tol
    end
end

function entropy(z::AbstractVector{Bool})::Float64
    zAz = EIGENDR_INPUT.A[z, z] |> sum
    zDz = EIGENDR_INPUT.D.diag[z] |> sum

    if isapprox(zDz, 0, atol = ENTROPY_PARAMS.tol) &&
       isapprox(zAz, 0, atol = ENTROPY_PARAMS.tol)
        return 0
    end
    if isapprox(zDz, 0, atol = ENTROPY_PARAMS.tol)
        if ENTROPY_PARAMS.topology
            return zAz * log2(zAz)
        else
            error("zDz is zero")
        end
    end
    if isapprox(zAz, 0, atol = ENTROPY_PARAMS.tol)
        return 0
    end
    if ENTROPY_PARAMS.topology
        return -zAz * log2(zAz) + 2 * zAz * log2(zDz)
    else
        return zAz * log2(zDz)
    end
end

function entropy(Z::BitMatrix)::Float64
    sum(entropy.(eachcol(Z)))
end

function entropy(l::AbstractVector{Int64})::Float64
    entropy(Z_from_labels(l))
end


# External
function entropy(
    z::AbstractVector{Bool},
    A::AbstractMatrix{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    D = Diagonal(sum(A, dims = 1)[:])
    zAz = A[z, z] |> sum
    zDz = D.diag[z] |> sum

    if isapprox(zDz, 0, atol = tol) && isapprox(zAz, 0, atol = tol)
        return 0
    end
    if isapprox(zDz, 0, atol = tol)
        if topology
            return zAz * log2(zAz)
        else
            error("zDz is zero")
        end
    end
    if isapprox(zAz, 0, atol = tol)
        return 0
    end
    if topology
        return -zAz * log2(zAz) + 2 * zAz * log2(zDz)
    else
        return zAz * log2(zDz)
    end
end

function entropy(
    Z::BitMatrix,
    A::AbstractMatrix{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    sum(entropy.(eachcol(Z), Ref(A), topology = topology, tol = tol))
end

function entropy(
    l::AbstractVector{Int64},
    A::AbstractMatrix{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    entropy(Z_from_labels(l), A, topology = topology, tol = tol)
end

end # module SE