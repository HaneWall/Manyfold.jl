"""

"""
struct DiffusionMap{T<:Real}
  d::Integer
  t::Integer
  α::T
  K::AbstractMatrix{T}
  Λs::AbstractMatrix{T}
  Vs::AbstractMatrix{T}
  Map::AbstractMatrix{T}
end


function fit(::Type{DiffusionMap}, X::AbstractMatrix{<:T}, kernel::Function;
  α=1.0, d::Integer=2, t::Integer=1, alg=:eigen, conj=false) where {T<:Real}
  K = pairwise(kernel, eachcol(X); symmetric=true)

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
  return DiffusionMap{T}(d, t, α, K, Λs, Vs, Map)
end


"""
How can we transform out of sample observation data X_new on the
diffusion coordinates.
"""
function transform(dmap::DiffusionMap, X_new::AbstractMatrix{<:T}; alg=:nystroem) where {T<:Real}
  if alg == :nystroem
    nothing
  end
end


"""
Lets say we have a new point Ψ_new, how can we get X_new?
Classic preimage problem.
"""
function preim()
  nothing
end

export fit, DiffusionMap
