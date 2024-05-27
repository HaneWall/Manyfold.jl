using Manyfold, Random

rng = MersenneTwister(1)
rng_oos = MersenneTwister(1234)

ε = 0.1

kernel = GaussianKernel(ε)

n = 6000
noise = 0.03
X, X_label = scurve(n, noise; segments=n, rng=rng)

n_oos = 100
X_oos, X_label_oos = scurve(n_oos, noise; segments=n_oos, rng=rng_oos)

dmap = fit(DiffusionMap, X, kernel; α=1.0, d=6, alg=:kry_eigen, conj=true)

psi_new = transform(dmap, X, X_oos)

fig_data = plot3D_data(X, X_label)
fig_matrix_diff = plot_matrix_coords(dmap.Map, 1, X_label)
fig_transform = plot_matrix_coords(psi_new, 1, X_label_oos)

figs = [fig_matrix_diff, fig_transform, fig_data]
save_figs(figs)
