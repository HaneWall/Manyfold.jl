using Manyfold, Random

rng = MersenneTwister(1)
ε = 0.25
kernel(x, y) = gaussian_kernel(x, y, ε)



n = 4000
noise = 0.03
X, X_label = scurve(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel; α=1.0, d=6, alg=:kry_eigen)

fig_data = plot3D_data(X, X_label)
fig_matrix_diff = plot_matrix_coords(dmap.Map, 1, X_label)
