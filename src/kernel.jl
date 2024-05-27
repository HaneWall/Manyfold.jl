using KernelFunctions

"""
    GaussianKernel(epsilon)

Creates KernelFunctions.Kernel object with lenghtscale epsilon: 
k(x, y) = exp(-d(x, y)^2 / (2 * epsilon)) with d(x, y) = |.|_2
"""
function GaussianKernel(epsilon::T) where {T<:Real}
  return KernelFunctions.with_lengthscale(KernelFunctions.SqExponentialKernel(), epsilon)
end


export GaussianKernel


