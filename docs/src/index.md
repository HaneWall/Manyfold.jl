````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Manyfold
  text: 
  tagline: Manifold learning with Diffusion Maps in Julia
  #image:
  #  src: filnename
  #  alt: filname alternative
  actions:
    - theme: brand
      text: Getting started
      link: 
    - theme: alt
      text: View on Github
      link: https://github.com/HaneWall/Manyfold.jl
---
````

Documentation for `Manyfold.jl`.

## Manifold Learning Package in Julia
Another manifold learning package in Julia. In this package 
we for now focus on nonlinear methods specifically the Diffusion Maps algorithm.
Might add other nonlinear dimensionality reduction methods in future versions to justify "Many"fold.

## Statement of Need:
1. missing / not working / wrong Diffusion Map algorithm in (no longer maintained?) ManifoldLearning.jl
2. first Julia Package to implement out of sample methods to map between Diffusion Map Coordinates and Ambient Space
3. provides lifting and restriction formalism to equation free methods with Diffusion Maps

## Implemented Diffusion Map Methods
- Nyström extension for restriction for out of sample data
- k-nearest neighbor method for lifting (using NearestNeighbor.jl) from latent space 
to ambient space 
- (Multi Scale) Geometric Harmonics for lifting and restriction

## Acknowledgements
- package is inspired by ManifoldLearning.jl and Datafold (wonderful Python package)

