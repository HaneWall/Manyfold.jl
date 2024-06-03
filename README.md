# Manyfold

[![Build Status](https://github.com/HaneWall/Manyfold.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HaneWall/Manyfold.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HaneWall/Manyfold.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HaneWall/Manyfold.jl)
## Manifold Learning Package in Julia
Another manifold learning package in Julia. In this package 
we for now focus on nonlinear methods specifically the Diffusion Maps algorithm.

## Statement of Need:
1. Missing / not working / wrong Diffusion Map Algorthm in (no longer maintained?) ManifoldLearning.jl
2. first Julia Package to implement out of sample methods to map between Diffusion Map Coordinates and Ambient Space
3. Lifting and Restriction formalism to "equation free methods"

## Implemented Diffusion Map Methods
- Nyström extension for restriction for out of sample data
- k-nearest neighbor method for lifting (using NearestNeighbor.jl) from latent space 
to ambient space 
- (Multi Scale) Geometric Harmonics for lifting and restriction
