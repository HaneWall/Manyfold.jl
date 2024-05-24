"""
    gaussian_kernel(x_i, x_j, eps::Real)

    eps ... width of gaussian (distance Obervation X)
    x_i, x_j ... columns of data matrix X

Creates a symmetric kernel matrix `K` based on a gaussian kernel and data values x ∈ X.
In applications we use StatsBase.pairwise() function that iterates over
the columns of the observation data X.
Since the gaussian_kernel is symmetric (as all diffusion kernels), we can
give the kwarg symmetric=true to pairwise in order to only compute the 
lower half of the kernel matrix and fill out the rest automatically.
"""
function gaussian_kernel(x_i, x_j, ε::Real)
  d² = sum((x_i - x_j) .^ 2)
  return exp(-d² / ε^2)
end

"""
    normalize_to_handle_density!(K::AbstractMatrix{<:Real}; α=0.0, symmetric=true)


α = 0.0 .. density has biggest influence (Graph Laplacian)
α = 0.5 .. Fokker Planck
α = 1.0 .. density is taken care of (Laplace-Beltrami)


Notice that if `K` is symmetric it does not matter if we sum over
the columns (dims=1) or rows (dims=2). However summation over columns is faster,
since Julia is column based.
"""
function normalize_to_handle_density!(K::AbstractMatrix{<:Real}; α=0.0)
  sums = sum(K, dims=2)
  P⁻ᵅ = Diagonal(1.0 ./ (vec(sums) .^ α))
  rmul!(lmul!(P⁻ᵅ, K), P⁻ᵅ)
end


"""
    normalize_to_right_stochastic!(P::AbstractMatrix{Real})

Takes a kernel matrix `K` and normalizes the rows to sum up to 1. 
Therefore we can reinterpret the matrix as a Markovian Matrix, 
that is right stochastic.
Notice that if `K` is symmetric it does not matter if we sum over
the columns (dims=1) or rows (dims=2). However summation over columns is faster,
since Julia is column based. Notice that we now lose 
"""
function normalize_to_right_stochastic!(K::AbstractMatrix{<:Real})
  sums = sum(K, dims=2)
  P⁻¹ = Diagonal(1.0 ./ vec(sums))
  lmul!(P⁻¹, K)
end


"""
    decompose(P::AbstractMatrix{<:Real}; skipfirst=true)

Decomposes the right/row stochastic matrix `P` into its eigenvalues
and eigenvecors. Since `P` is stochastic the first eigenvalue is 1 and 
the first eigenvector is the trivial one, therefore we skip them. 

We return a diagonal matrix Λs containing descending eigenvalues and 
corresponding columnwise ordered eigenvectors matrix Vs.
"""
function decompose(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst=true, alg=:kry_eigen)
  if alg == :eigen
    Λs, Vs = eigsolver(P, dim; skipfirst)
  elseif alg == :svd
    Λs, Vs = svdsolver(P, dim; skipfirst)
  elseif alg == :kry_eigen
    Λs, Vs = kry_eigsolver(P, dim; skipfirst)
  elseif alg == :kry_svd
    Λs, Vs = kry_svdsolver(P, dim; skipfirst)
  end
  return Λs, Vs
end

function eigsolver(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst)
  eigen_decomposition = eigen(P)
  eig_vals = eigen_decomposition.values
  eig_vecs = eigen_decomposition.vectors
  # in order to sort eigvals we have to neglect the imaginary part
  # which can be introduced due to numerical errors 

  for val in eig_vals
    if isa(val, Complex)
      eig_vals = real(eig_vals)
    end
  end

  if skipfirst
    order_idx = sortperm(Float64.(eig_vals), rev=true)[2:dim+1]
  else
    order_idx = sortperm(Float64.(eig_vals), rev=true)[1:dim]
  end
  Λs = Diagonal(eig_vals[order_idx])
  Vs = real.(eig_vecs[:, order_idx])
  return Λs, Vs
end

function kry_eigsolver(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst)
  λs, vs, _ = KrylovKit.eigsolve(P, dim + 1, :LM)
  l = size(P)[1]
  Vs = zeros(Float64, l, dim)
  idx_s = skipfirst ? range(2, dim + 1) : range(1, dim)
  for (idx_m, idx) in enumerate(idx_s)
    λs[idx] = real(λs[idx])
    Vs[:, idx_m] .= real.(vs[idx])
  end
  Λs = Diagonal(vec(λs[idx_s]))
  return Λs, Vs
end

function svdsolver(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst)
  U, s, _ = svd(P)
  if skipfirst
    return Diagonal(vec(s)[2:dim+1]), U[:, 2:dim+1]
  else
    return Diagonal(vec(s)[1:dim]), U[:, 1:dim]
  end
end

function kry_svdsolver(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst)
  σs, lvecs, rvecs, _ = KrylovKit.svdsolve(P, dim + 1, :LR)
  l = size(P)[1]
  Vs = zeros(Float64, l, dim)
  idx_s = skipfirst ? range(2, dim + 1) : range(1, dim)
  for (idx_m, idx) in enumerate(idx_s)
    Vs[:, idx_m] .= lvecs[idx]
  end
  Λs = Diagonal(vec(σs[idx_s]))
  return Λs, Vs
end

export gaussian_kernel, normalize_to_right_stochastic!, normalize_to_handle_density!, decompose
