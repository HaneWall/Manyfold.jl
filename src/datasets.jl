"""
  Swiss roll from ManifoldLearning.jl
"""
function swiss_roll(n::Int=1000, noise::Real=0.03; segments=1, hlims=(-10.0, 10.0),
  rng::AbstractRNG=default_rng())
  t = (3 * pi / 2) * (1 .+ 2 * rand(rng, n, 1))
  height = (hlims[2] - hlims[1]) * rand(rng, n, 1) .+ hlims[1]
  X = [t .* cos.(t) height t .* sin.(t)]
  X .+= noise * randn(rng, n, 3)
  mn, mx = extrema(t)
  labels = segments == 0 ? t : round.(Int, (t .- mn) ./ (mx - mn) .* (segments - 1))
  return collect(transpose(X)), vec(labels)
end

"""
  scurve from ManifoldLearning.jl
"""
function scurve(n::Int=1000, noise::Real=0.03; segments=1,
  rng::AbstractRNG=default_rng())
  t = 3π * (rand(rng, n) .- 0.5)
  x = sin.(t)
  y = 2rand(rng, n)
  z = sign.(t) .* (cos.(t) .- 1)
  height = 30 * rand(rng, n, 1)
  X = [x y z] + noise * randn(n, 3)
  mn, mx = extrema(t)
  labels = segments == 0 ? t : round.(Int, (t .- mn) ./ (mx - mn) .* (segments - 1))
  return collect(transpose(X)), vec(labels)
end


export swiss_roll, scurve

