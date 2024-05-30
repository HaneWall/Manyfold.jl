"""
    normalize_to_handle_density!(K::AbstractMatrix{<:Real}; α=0.0, symmetric=true)


α = 0.0 .. density has biggest influence (Graph Laplacian)
α = 0.5 .. Fokker Planck
α = 1.0 .. density is taken care of (Laplace-Beltrami)

"""
function normalize_to_handle_density!(K::AbstractMatrix{<:Real}; α=0.0)
  sums = sum(K, dims=2)
  P⁻ᵅ = Diagonal(1.0 ./ (vec(sums))) .^ α
  rmul!(lmul!(P⁻ᵅ, K), P⁻ᵅ)
end


"""
    normalize_to_right_stochastic!(P::AbstractMatrix{Real})

Takes a kernel matrix `K` and normalizes the rows to sum up to 1. 
Therefore we can reinterpret the matrix as a Markovian Matrix, 
that is right stochastic.
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

We return a diagonal matrix Λs containing descending eigenvalues/singularvalues and 
corresponding columnwise ordered eigenvectors matrix Vs.
"""
function decompose(P::AbstractMatrix{<:Real}, dim::Integer;
  skipfirst=true, alg=:kry_eigen)
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



"""
    decompose_sym(K::AbstractMatrix{<:Real}, dim::Intger; skipfirst=true, alg=:kry_eigen)
Introduces a change of basis in order for symmetrification of the kernelmatrix `K`.
The new matrix has the same eigenvalues
"""
function decompose_sym(K::AbstractMatrix{<:Real}, dim::Integer;
  skipfirst=true, alg=:kry_svd)
  sums = sum(K, dims=2)
  P = sqrt.(Diagonal(1.0 ./ vec(sums)))
  rmul!(lmul!(P, K), P)
  K_sym = Symmetric(K)
  if alg == :eigen
    Λs, Vs = eigsolver(K_sym, dim; skipfirst)
  elseif alg == :svd
    Λs, Vs = svdsolver(K_sym, dim; skipfirst)
  elseif alg == :kry_eigen
    Λs, Vs = kry_eigsolver(K_sym, dim; skipfirst)
  elseif alg == :kry_svd
    Λs, Vs = kry_svdsolver(K_sym, dim; skipfirst)
  end
  lmul!(P, Vs)
  return Λs, Vs
end

function eigsolver(P::AbstractMatrix{<:Real}, dim::Integer; skipfirst)
  eigen_decomposition = eigen(P, sortby=abs)
  eig_vals = eigen_decomposition.values
  eig_vecs = eigen_decomposition.vectors
  # in order to sort eigvals we have to neglect the imaginary part
  # which can be introduced due to numerical errors 

  eig_vals = real.(eig_vals)

  if skipfirst
    eig_vals = reverse(eig_vals)[2:dim+1]
    Vs = real.(reverse(eig_vecs, dims=2))[:, 2:dim+1]
  else
    eig_vals = reverse(eig_vals)[1:dim]
    Vs = real.(reverse(eig_vecs, dims=2))[:, 1:dim]
  end
  Λs = Diagonal(eig_vals)
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
  σs, lvecs, _ = KrylovKit.svdsolve(P, dim + 1, :LR)
  l = size(P)[1]
  Vs = zeros(Float64, l, dim)
  idx_s = skipfirst ? range(2, dim + 1) : range(1, dim)
  for (idx_m, idx) in enumerate(idx_s)
    Vs[:, idx_m] .= lvecs[idx]
  end
  Λs = Diagonal(vec(σs[idx_s]))
  return Λs, Vs
end

export gaussian_kernel, normalize_to_right_stochastic!, normalize_to_handle_density!, decompose, decompose_sym
