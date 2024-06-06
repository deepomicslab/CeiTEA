module StructuralEntropy

export se

using ..Utils: Z_from_labels

using LinearAlgebra: Diagonal

function se(
    z::AbstractVector{Bool},
    A::Matrix{Float64},
    D::Diagonal{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    # zAz = z' * A * z
    # zDz = z' * D * z
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

function se(
    Z::BitMatrix,
    A::Matrix{Float64},
    D::Diagonal{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    sum(se.(eachcol(Z), Ref(A), Ref(D), topology = topology, tol = tol))
end

function se(
    l::AbstractVector{Int64},
    A::Matrix{Float64},
    D::Diagonal{Float64};
    topology = true,
    tol = eps(Float32),
)::Float64
    Z = Z_from_labels(l)
    se(Z, A, D; topology = topology, tol = tol)
end

end # module SE