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
  k::Kernel
  K_m::AbstractMatrix{T}
  K::AbstractMatrix{T}
  Λs::AbstractMatrix{T}
  Vs::AbstractMatrix{T}
  Map::AbstractMatrix{T}
end

"""


"""
function fit(::Type{DiffusionMap}, X::AbstractMatrix{<:T}, kernel::S;
  α=1.0, d::Integer=2, t::Integer=1, alg=:eigen, conj=false) where {S<:Kernel,T<:Real}
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
  Map = Λs^t * Vs'
  return DiffusionMap{T}(d, t, α, kernel, K_m, K, Λs, Vs, Map)
end


"""
How can we transform out of sample observation data X_new on the
diffusion coordinates.
"""
function transform(dmap::DiffusionMap, X_train::AbstractMatrix{T}, X_oos::AbstractMatrix{T}; alg=:nystrom) where {T<:Real}
  if alg == :nystrom
    K_new = KernelFunctions.kernelmatrix(dmap.k, ColVecs(X_oos), ColVecs(X_train))
    if !(isapprox(dmap.α, 0.0))
      p⁻ᵅ = Diagonal(1.0 ./ (vec(sum(K_new, dims=2)))) .^ (dmap.α)
      lmul!(p⁻ᵅ, K_new)
      P⁻ᵅ = Diagonal(1.0 ./ (vec(sum(dmap.K_m, dims=2)))) .^ (dmap.α)
      rmul!(K_new, P⁻ᵅ)
    end
    normalize_to_right_stochastic!(K_new)
    Ψ = zeros(T, dmap.d, size(X_oos)[2])
    # for dmap_idx in axes(Ψ, 1)
    #   for oos_idx in axes(Ψ, 2)
    #     Ψ[dmap_idx, oos_idx] = 1 / dmap.Λs[dmap_idx, dmap_idx]^dmap.t .* dot(K_new[oos_idx, :], dmap.Vs[:, dmap_idx])
    #   end
    # end
    Ψ = transpose(K_new * dmap.Vs * dmap.Λs^(-1 * dmap.t))
    return Ψ
  end
end


function transform(dmap::DiffusionMap, X_train::AbstractMatrix{T}, X_oos::AbstractVector{T}; alg=:nystrom) where {T<:Real}
  return transform(dmap, X_train, reshape(X_oos, size(X_train)[1], 1); alg)
end

"""
Lets say we have a new point Ψ_new, how can we get X_new?
Classic preimage problem.
"""
function preim()
  nothing
end

export fit, DiffusionMap, transform
