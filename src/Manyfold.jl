module Manyfold

using CairoMakie, LinearAlgebra, StatsBase, Random, KrylovKit, KernelFunctions

include("DMaps.jl")
include("utils.jl")
include("datasets.jl")
include("styles.jl")
include("kernel.jl")
end
