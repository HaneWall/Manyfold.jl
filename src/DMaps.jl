"""
    DiffusionMap

Struct that stores various informations about the diffusion map. 
    `d` ... number of dimensions
    `t` ... timesteps of diffusion process
    `α` ... exponent that handles density of data
    `k` ... kernel function, that determines law to gain KernelMatrix
    `K_m` ... KernelKatrix (kernelfunction applied on Data)
    `K` ... modified Matrix to detrmine diffusion map coordinates
    `Λs` ... Diagonal matrix of descending eigenvalues/singularvalues of `K`
    `Vs` ... Eigenvectors column wise
    `Map` ... DiffusionMap coordinates (column wise)
"""
struct DiffusionMap{T<:Real}
  d::Integer
  t::Integer
  α::T
  X_train::AbstractMatrix{T}
  k::Kernel
  # following quantities will we determined and are not free to choose
  K_m::AbstractMatrix{T}
  K::AbstractMatrix{T}
  Λs::AbstractMatrix{T}
  Vs::AbstractMatrix{T}
  Map::AbstractMatrix{T}
end

"""


"""
function fit(::Type{DiffusionMap}, X::AbstractMatrix{<:T}, kernel::S;
  α=1.0, d::Integer=2, t::Integer=1, alg=:kry_eigen, conj=false) where {S<:Kernel,T<:Real}
  K_m = KernelFunctions.kernelmatrix(kernel, ColVecs(X))
  K = copy(K_m)
  if !(isapprox(α, 0))
    normalize_to_handle_density!(K; α)
  end
  if !conj
    normalize_to_right_stochastic!(K)
    Λs, Vs = decompose(K, d; alg=alg)
  else
    Λs, Vs = decompose_sym(K, d)
  end
  Map = Vs * Λs^t
  return DiffusionMap{T}(d, t, α, X, kernel, K_m, K, Λs, Vs, Map)
end


"""
How can we transform out of sample observation data X_new on the
diffusion coordinates.
"""
function transform(dmap::DiffusionMap, X_oos::AbstractMatrix{T}; alg=:nystrom) where {T<:Real}
  if alg == :nystrom
    K_new = KernelFunctions.kernelmatrix(dmap.k, ColVecs(X_oos), ColVecs(dmap.X_train))
    if !(isapprox(dmap.α, 0.0))
      p⁻ᵅ = Diagonal(1.0 ./ (vec(sum(K_new, dims=2)))) .^ (dmap.α)
      lmul!(p⁻ᵅ, K_new)
      P⁻ᵅ = Diagonal(1.0 ./ (vec(sum(dmap.K_m, dims=2)))) .^ (dmap.α)
      rmul!(K_new, P⁻ᵅ)
    end
    normalize_to_right_stochastic!(K_new)
    Y = K_new * dmap.Vs * dmap.Λs^(-1 * dmap.t)
    return Y
  end
end


function transform(dmap::DiffusionMap, X_oos::AbstractVector{T}; alg=:nystrom) where {T<:Real}
  return transform(dmap, dmap.X_train, reshape(X_oos, size(dmap.X_train)[1], 1); alg)
end

"""
Lets say we have a new points Y_oos in the low
dimensional latent space, how can we get X_new in the
high dimesional ambient space?
This is the classic ill-defined preimage problem.
"""
function preim_double_diffusionmap(dmap::DiffusionMap, Y_oos::AbstractMatrix{T};
  kernel::S=dmap.k, n_eigenpairs::Integer=6, alg=:latent_harmonic) where {S<:Kernel,T<:Real}
  doubledmap = fit(DiffusionMap, Y_oos, kernel; α=1.0, t=1, alg=:kry_eigen, conj=true, d=n_eigenpairs)
end


"""
    preim_knn(dmap::DiffusionMap, Y_oos::AbstractMatrix{T}; k=8)
dmap ... DiffusionMap object
Y_oos ... n out of sample points in the latent space Y_oos_i ∈ R ^ d
d ... dimension latent space
k ... number of nearest neighbors that we search for in the latent space.

Generally speaking we search for the k nearest neighbors in the d-dimensional 
latent space for each of the n-samples. From these k nearest neighbors we 
can look at their respective high dimensional data in the ambient space

This approach is for example used by Sandstede [https://arxiv.org/pdf/2112.15159]
"""
function preim_knn(dmap::DiffusionMap, Y_oos::AbstractMatrix{T}; k::Integer=8) where {T<:Real}
  d, n = size(Y_oos)
  D, N = size(dmap.X_train)
  knn_idxs, knn_points = NearestNeighbors.knn(KDTree(dmap.Map), Y_oos, k, true)
  ambient_knn_points_cube = [@views dmap.X_train[:, idx_set] for idx_set in knn_idxs]
  # initial guess of the coefficents is just the arithmetic mean
  coeff_matrix = 1 / k * ones(T, n, k)
  preim_Y_oos = zeros(T, D, n)
  for (idx_points, ambient_knn_points_matrix) in enumerate(ambient_knn_points_cube)
    for (idx, point) in enumerate(eachcol(ambient_knn_points_matrix))
      preim_Y_oos[:, idx_points] += coeff_matrix[idx, :] * point
    end
  end
  return preim_Y_oos
end


"""
preim_knn(dmap::DiffusionMap, Y_oos, k)
"""
function preim_knn(dmap::DiffusionMap, Y_oos::AbstractVector{T}; k::Integer=8) where {T<:Real}
  d = length(Y_oos)
  knn_idxs, knn_points = NearestNeighbors.knn(KDTree(dmap.Map), Y_oos, k, true)
  ambient_points = [@views dmap.X_train[:, knn_idxs]]
end

export fit, DiffusionMap, transform
