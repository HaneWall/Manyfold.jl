"""
    GeometricHarmonics{T<:Real}
Is an object that allows us jump between the ambient space and latent space.
From the "restriction" standpoint this is an alternative to the natural 
Nyström extension. From the "lifting" standpoint this is an alternative
to k-nearest neighbors.

We do not care about the direction here (restriction or lifting)!
In later prediction processes we always map the following way:
GH: X-->Y.
Generally speaking GeometricHarmonics were introduced by Coifman himself. 
Over the years many adaptations arised and nowadays GeometricHarmonics can also 
be used to lift from the embedded latent space to the high dimensional ambient space 
via so-called Latent Harmonics (double diffusion Map).

d ... dimensions of the eigenspace (usually one has to take into account many)
α ... renormalization parameter ∈ [0, 1]
X_train ... training data from the domain
Y_train ... training data of the image(X_train)
k ... kernel function that is used to create kernelmatrices
K ... kernelmatrix
Λs ... Eigenvalues of K (descending order in diagonal matrix form)
Vs ... Eigenvectors in column wise order corresponding to Λs
Map ... Y_train map projections on GeometricHarmonics
"""
struct GeometricHarmonics{T<:Real}
  d::Integer
  α::T
  X_train::AbstractMatrix{T}
  Y_train::AbstractMatrix{T}
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
function fit(::Type{GeometricHarmonics}, X_train::AbstractMatrix{T}, Y_train::AbstractMatrix{T}, kernel::S;
  α::Real=1.0, d=10, alg=:eigen) where {S<:Kernel,T<:Real}
  K_new = KernelFunctions.kernelmatrix(kernel, ColVecs(X_train))
  if !(isapprox(α, 0))
    p⁻ᵅ = Diagonal(1.0 ./ (vec(sum(K_new, dims=2)))) .^ (α)
    lmul!(p⁻ᵅ, K_new)
    normalize_to_right_stochastic!(K_new)
  end
  #the first eigenvector/value is important in this case
  Λs, Vs = decompose(K_new, d; skipfirst=false, alg)
  Map = Vs * Λs^(-1) * transpose(Vs) * Y_train
  return GeometricHarmonics{T}(d, α, X_train, Y_train, kernel, K_new, Λs, Vs, Map)
end


"""
    predict(GH::GeometricHarmonics, X_oos)
Uses the learned GeometricHarmonics model to fit new out of sample data X_oos to predict 
on Y living space.
"""
function predict(GH::GeometricHarmonics, X_oos)
  K_new = KernelFunctions.kernelmatrix(GH.k, ColVecs(X_oos), ColVecs(GH.X_train))
  if !(isapprox(GH.α, 0))
    normalize_to_right_stochastic!(K_new)
  end
  return K_new * GH.Map
end


struct MultiScaleGeometricHarmonics{T<:Real}
  d::Integer
  α::T
  X_train::AbstractMatrix{T}
  Y_train::AbstractMatrix{T}
  k::Kernel
  # following quantities will we determined and are not free to choose
  K::AbstractMatrix{T}
  Λs::AbstractMatrix{T}
  Vs::AbstractMatrix{T}
  Map::AbstractMatrix{T}
end

export GeometricHarmonics, fit, predict
