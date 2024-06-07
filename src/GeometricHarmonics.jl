"""
    struct  GeometricHarmonics{T<:Real}
Is an object that allows us to jump between the ambient and latent space.
From the "restriction" standpoint this is an alternative to the natural
Nyström extension. From the "lifting" standpoint this is an alternative
to the natural k-nearest neighbors approach.
For details look up [Coifman](https://doi.org/10.1016/j.acha.2005.07.005) and [Kevrekedis](https://doi.org/10.1016/j.jcp.2023.112072).
Direction for predictions: X --> Y

# Fields
  - `d` : dimensions of the eigenspace (usually one has to take into account many)
  - `X_train` : training data from the domain 
  - `Y_train` : training data of image(`X_train`)
  - `k` : kernel function that is used to create kernelmatrices
  - `K` : kernelmatrix
  - `Λs` : eigenvalues of `K` in diagonal matrix (descending order)
  - `Vs` : eigenvectors in column wise order corresponding to `Λs`
  - `Map` :  `Y_train` projections on geometric harmonics
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
    fit(::Type{GeometricHarmonics}, 
        X_train::AbstractMatrix{T}, 
        Y_train::AbstractMatrix{T}, 
        kernel::S; kwargs...) where {T<:Real}

Creates GeometricHarmonics object, that is oriented to map from `X_train` to `Y_train`.
Cann be used for Lifting and Restriction.

# Arguments
  - `::Type{GeometricHarmonics}` : declares that we want to create GeometricHarmonics object
  - `X_train` : training data from the domain
  - `Y_train` : training data of image(`X_train`)
  - `kernel` : kernelfunction

# Keyword Arguments
  - `d=10` : number of eigenvectors that we would like to take into account
  - `alg=:eigen` : eigensolver backend. Options: `:eigen`, `:svd`
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

Uses the learned GeometricHarmonics model to fit new 
out of sample data X_oos to predict on Y_train living space.

# Arguments
  - `GH` : already trained `GeometricHarmonics` object
  - `X_oos` : out of sample new data set. Rows: features, Cols: samples
"""
function predict(GH::GeometricHarmonics, X_oos::AbstractMatrix{T}) where {T<:Real}
  K_new = KernelFunctions.kernelmatrix(GH.k, ColVecs(X_oos), ColVecs(GH.X_train))
  return K_new * GH.Map
end


"""
    struct MultiScaleGeometricHarmonics

WORK IN PROGRESS
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
