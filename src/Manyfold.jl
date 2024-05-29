module Manyfold

using CairoMakie, LinearAlgebra, StatsBase
using Random, KrylovKit, KernelFunctions
using NearestNeighbors

include("DMaps.jl")
include("utils.jl")
include("datasets.jl")
include("styles.jl")
include("kernel.jl")
end
