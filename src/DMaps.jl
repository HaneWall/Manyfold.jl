"""
    struct  DiffusionMap{T<:Real}

Struct that stores various informations about the diffusion map.
# Fields
  - `d` : number of dimensions
  - `t` : timesteps of diffusion process
  - `α` : exponent that handles density of data
  - `k` : kernel function, that determines law to gain KernelMatrix
  - `K_m` : KernelKatrix (kernelfunction applied on Data)
  - `K` : modified Matrix to detrmine diffusion map coordinates
  - `Λs` :  Diagonal matrix of descending eigenvalues/singularvalues of `K`
  - `Vs` : eigenvectors column wise
  - `Map` : DiffusionMap coordinates (column wise)
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
    fit(::Type{DiffusionMap}, X::AbstractMatrix{<:T}, 
        kernel::S; kwargs...) where {S <: Kernel, T <:Real}

Given data set `X` we compute diffusion map coordinates with a given kernelfunction `kernel`.
# Arguments
  - `::Type{DiffusionMap}` : declares that we use Diffusion Maps to fit data
  - `X` : data matrix, features represent rows, whereas columns represent different samples
  - `kernel` : kernel function, that is used to produce kernel matrix

# Keyword Arguments
  - `α=1.0` : density of points regulator. 0.0 -> Graph Laplacian, 0.5 -> Fokker Planck, 1.0 -> Laplace Beltrami
  - `d=2` : number of dimensions / eigenvectors that we would like to take into account (ordered by largest eigenvalues)
  - `t=1` : timesteps of diffusion
  - `alg=:kry_eigen` : eigen-/svdproblem solver. Options: `:eigen`, `:svd`, `:kry_eigen`, `:kry_svd`
  - `conj=false` : if true we symmetrize the kernel matrix by proper transformation and can use svd solver
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
    transform(dmap::DiffusionMap, X_oos::AbstractMatrix{T}; kwargs...) where {T<:Real}
Given out of sample data set `X_oos` we compute embedding `Y_oos`
in the diffusion map coordinates by the Nystrom extension.

# Arguments
  - `dmap` : Diffusion map object that was created beforehand with training data
  - `X_oos` : oout of sample data set. Rows represent features, columns represent samples.

# Keyword Arguments
  - `alg=:nystrom` : method to transform new out of sample data
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
    preim_knn(dmap::DiffusionMap, Y_oos::AbstractMatrix{T}; kwargs...)

Solves preimage problem for out of sample data set `Y_oos` in the diffusion map coordinates
via k-nearest neighbor approach. This approach is for example used by [Sandstede 2021](https://arxiv.org/pdf/2112.15159)
We handle this problem by an divide and conquer approach for readibility.
# Arguments
  - `dmap` : Diffusion map object that was created beforehand with training data
  - `Y_oos` : out of sample points in the latent space. Rows: Features, Cols: Samples

# Keyword Arguments
  - `k=8` : number of nearest neighbors
  - `alg=:arithmetic` : algorithm to get higher dimensional point. Options: `:arithmetic`, `:conv_hull`
  - `embed_dim` : If `Y_oos` is not defined on all dimensions of the diffusion map, i.e.
    "[1 5] -> `Y_oos`" lives on first and fifth coordinate
"""
function preim_knn(dmap::DiffusionMap, Y_oos::AbstractMatrix{T};
  k::Integer=8, embed_dim::AbstractVector{S}=collect(Integer, 1:size(dmap.Map)[2]),
  alg=:arithmetic) where {S<:Integer,T<:Real}

  preim_Y_oos = zeros(T, size(dmap.X_train)[1], size(Y_oos)[2])
  #TODO there is free food here for parallelization
  for (point_idx, Y_oos_point) in enumerate(eachcol(Y_oos))
    preim_Y_oos[:, point_idx] .= preim_knn(dmap, Y_oos_point; k, embed_dim, alg)
  end
  return preim_Y_oos
end


function preim_knn(dmap::DiffusionMap, Y_oos::AbstractVector{T};
  k::Integer=8, embed_dim::AbstractVector{S}=collect(Integer, 1:size(dmap.Map)[2]),
  alg=:arithmetic) where {S<:Integer,T<:Real}

  knn_idxs, _ = NearestNeighbors.knn(BruteTree(dmap.Map[:, embed_dim]'), Y_oos, k, true)
  ambient_points = @views dmap.X_train[:, knn_idxs]
  #initial guess of the coefficients is the arithmetic mean
  coeffs = 1 / k * ones(T, k)
  preim_Y_oos_point = zeros(T, size(dmap.X_train)[1])
  #for now we just take the arithmetic mean, later optimization problem
  if alg == :arithmetic
    preim_Y_oos_point = mean(ambient_points, dims=2)
  elseif alg == :conv_hull
    #TODO implement optimization problem to get coeffs
    for (idx_ambient_point, ambient_point) in enumerate(eachcol(ambient_points))
      preim_Y_oos_point .= preim_Y_oos_point .+ coeffs[idx_ambient_point] .* ambient_point
    end
  else
    nothing
  end
  return preim_Y_oos_point
end

# function _knn_convex_hull_min(dmap::DiffusionMap, coeffs::AbstractVector{T}, ambient_points::AbstractMatrix(T), preim_)
#   nothing
# end


export fit, DiffusionMap, transform, preim_knn
