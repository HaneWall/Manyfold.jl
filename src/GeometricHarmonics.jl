struct GeometricHarmonics{T<:Real}
  d::Integer
  α::T
  X_train::AbstractMatrix{T}
  Y_target::AbstractMatrix{T}
  k::Kernel
  # following quantities will we determined and are not free to choose
  K::AbstractMatrix{T}
  Λs::AbstractMatrix{T}
  Vs::AbstractMatrix{T}
  Map::AbstractMatrix{T}
end


"""
    fit()

Fits a GeometricHarmonics model to interpolate a map from X_train data space to Y_target data space.
This approach does not really care about the dimensions and can be used for Lifting and Restriction.
However one should be careful at comparing the restriction to Nyström, since we only restrict to 
the choosen embedding but not all eigenfunctions of the diffusion map.
"""
function fit(::Type{GeometricHarmonics}, X_train::AbstractMatrix{T}, Y_target::AbstractMatrix{T}, kernel::S;
  α::Real=1.0, conj=false, d=10, alg=:kry_eigen) where {S<:Kernel,T<:Real}
  K = KernelFunctions.kernelmatrix(kernel, ColVecs(X_train))
  if !(isapprox(α, 0))
    normalize_to_handle_density!(K; α)
  end
  if !conj
    normalize_to_right_stochastic!(K)
    Λs, Vs = decompose(K, d; alg=alg)
  else
    Λs, Vs = decompose_sym(K, d)
  end
  Map = Vs * Λs^(-1) * transpose(Vs) * Y_target
  return GeometricHarmonics{T}(d, α, X_train, Y_target, kernel, K, Λs, Vs, Map)
end


"""
    predict(GH::GeometricHarmonics, X_oos)
Uses the learned GeometricHarmonics model to fit new out of sample data X_oos to predict 
on Y_target living space
"""
function predict(GH::GeometricHarmonics, X_oos)
  K_new = KernelFunctions.kernelmatrix(GH.k, ColVecs(X_oos), ColVecs(GH.X_train))
  normalize_to_right_stochastic!(K_new)
  return K_new * GH.Map
end

export GeometricHarmonics, fit, predict