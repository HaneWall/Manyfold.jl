"""
    struct  GeometricHarmonics{T<:Real}
using CairoMakie: project_polygon
Is an object that allows us to jump between the ambient and latent space.
From the "restriction" standpoint this is an alternative to the natural
Nyström extension. From the "lifting" standpoint this is an alternative
to the k-nearest neighbors approach.

We do not care about the direction here (restriction or lifting)!
In later prediction processes we always map the following way:
GH: F_1^(D_1 × N) --> F_2(D_2 × N), X_train |-> Y_train
(N-samples are column wise)
Generally speaking GeometricHarmonics were introduced by Coifman himself. 
[https://doi.org/10.1016/j.acha.2005.07.005]
Over the years many adaptations arised and nowadays GeometricHarmonics can also 
be used to lift from the embedded latent space to the high dimensional ambient space 
via so-called Latent Harmonics (double diffusion Map)
[https://doi.org/10.1016/j.jcp.2023.112072]


d ... dimensions of the eigenspace (usually one has to take into account many)
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

Fits a GeometricHarmonics model to interpolate a map from X_train data space to Y_train data space.
This approach does not really care about the dimensions and can be used for Lifting and Restriction.
However one should be careful at comparing the restriction to Nyström, since we only restrict to
the choosen embedding but not all eigenfunctions of the diffusion map.
"""
function fit(::Type{GeometricHarmonics}, X_train::AbstractMatrix{T}, Y_train::AbstractMatrix{T}, kernel::S;
  d::Integer=10, alg=:eigen) where {S<:Kernel,T<:Real}
  K_new = KernelFunctions.kernelmatrix(kernel, ColVecs(X_train))
  #the first eigenvector/value is important in this case
  Λs, Vs = decompose(K_new, d; skipfirst=false, alg)
  Map = Vs * Λs^(-1) * transpose(Vs) * Y_train
  return GeometricHarmonics{T}(d, X_train, Y_train, kernel, K_new, Λs, Vs, Map)
end


""" 
    predict(GH::GeometricHarmonics, X_oos)
Uses the learned GeometricHarmonics model to fit new out of sample data X_oos to predict 
on Y_train living space.
"""
function predict(GH::GeometricHarmonics, X_oos::AbstractMatrix{T}) where {T<:Real}
  K_new = KernelFunctions.kernelmatrix(GH.k, ColVecs(X_oos), ColVecs(GH.X_train))
  return K_new * GH.Map
end


"""
    struct MultiScaleGeometricHarmonics

Introduction of multiscale GeometricHarmonics. General idea is to
incorparate multiple length scales / band widths ε into our kernel.
Therefore we get better approximations for the extension of the domain.
"""
struct MultiScaleGeometricHarmonics{T<:Real}
  X_train::AbstractMatrix{T}
  Y_train::AbstractMatrix{T}
  error::T
  δ::T
  ε::T
  # following quantities will we determined and are not free to choose
  ks::AbstractArray{<:Kernel}
  Maps::AbstractArray{T}
end

"""
    fit(::Type{MultiScaleGeometricHarmonics}, X_train::AbstractMatrix{T}, Y_train::AbstractMatrix{T};
  δ::T=1e-5, ε_init::T=1.0, error::T=1e-10, l_max::Integer=7, μ::T=2.0, alg=:eigen) where {T<:Real}

TBW
"""
function fit(::Type{MultiScaleGeometricHarmonics}, X_train::AbstractMatrix{T}, Y_train::AbstractMatrix{T};
  δ::T=1e-5, ε_init::T=1.0, error::T=1e-10, l_max::Integer=7, μ::T=2.0, alg=:eigen) where {T<:Real}
  # for now empty array of kernelfunctions
  ks = KernelFunctions.Kernel[]
  Maps = zeros(T, size(Y_train)[1], size(Y_train)[2], l_max)
  ε = ε_init
  Y_rolling_approx = copy(Y_train)
  for l in 1:l_max
    k_new = KernelFunctions.with_lengthscale(KernelFunctions.SqExponentialKernel(), ε)
    push!(ks, k_new)
    Map = @views Maps[:, :, l]
    K_mat = kernelmatrix(k_new, ColVecs(X_train))
    Λs, Vs = decompose_δ(K_mat, δ; skipfirst=false, alg)
    proj_y = Vs * transpose(Vs) * Y_rolling_approx
    Map .= Vs * Λs^(-1) * transpose(Vs) * Y_rolling_approx
    Y_rolling_approx .-= proj_y
    ε /= μ
  end
  return MultiScaleGeometricHarmonics{T}(X_train, Y_train, error, δ, ε_init, ks, Maps)
end

function predict(MGH::MultiScaleGeometricHarmonics, X_oos::AbstractMatrix{T}) where {T<:Real}
  Y_oos = zeros(T, size(X_oos)[2], size(MGH.Y_train)[2])
  for (idx, k) in enumerate(MGH.ks)
    K_mat = KernelFunctions.kernelmatrix(k, ColVecs(X_oos), ColVecs(MGH.X_train))
    Y_oos .+= K_mat * MGH.Maps[:, :, idx]
  end
  return Y_oos
end

export GeometricHarmonics, MultiScaleGeometricHarmonics, fit, predict
