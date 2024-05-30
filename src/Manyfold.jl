module Manyfold

using CairoMakie, LinearAlgebra, StatsBase
using Random, KrylovKit, KernelFunctions
using NearestNeighbors, GLMakie

include("DMaps.jl")
include("GeometricHarmonics.jl")
include("utils.jl")
include("datasets.jl")
include("styles.jl")
include("kernel.jl")
end
