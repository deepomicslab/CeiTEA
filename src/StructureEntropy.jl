module StructureEntropy

using LinearAlgebra

# Write documentation for the following function
"""
    quad_prod(v, M)

Compute the quadratic form of a matrix `M`.

# Arguments
- `v::AbstractVector`: The vector.
- `M::AbstractMatrix`: The matrix.

# Example
```julia
M = [1 2; 3 4]
v = [1, 2]
quad_prod(v, M)
```
"""
function quad_prod(v, M)
    v' * M * v
end

"""
    se(x, A)

Compute the structure entropy of a matrix `A` with respect to an indicator vector `x`.

# Arguments
- `x::AbstractMatrix`: The vector.
- `A::AbstractMatrix`: The matrix.

# Example
```julia
A = [1 2; 3 4]
x = [1, 2]
se(x, A)
```
"""
function se(x, A)
    D = Diagonal(sum(A, dims=1)[:])
    sum(quad_prod(x[:, i], A) * log2(quad_prod(x[:, i], D)) for i in axes(x, 2))
end

export quad_prod, se

end # module StructureEntropy