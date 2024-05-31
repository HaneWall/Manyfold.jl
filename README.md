# Manyfold

[![Build Status](https://github.com/HaneWall/Manyfold.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaneWall/Manyfold.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HaneWall/Manyfold.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HaneWall/Manyfold.jl)
## Manifold Learning Package in Julia

Yet another manifold learning package in Julia. In this package 
we for now focus on nonlinear methods specifically the Diffusion Maps algorithm.

## Implemented Diffusion Map Methods
- Nyström extension for restriction for out of sample data
- k-nearest neighbor method for lifting (using NearestNeighbor.jl) from latent space 
to ambient space 
- Geometric Harmonics for lifting as well as restriction
