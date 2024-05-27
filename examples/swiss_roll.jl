using Manyfold
using Random


rng = MersenneTwister(1)
ε = 1.0

kernel = GaussianKernel(ε)

n = 7000
noise = 0.2
X, X_label = swiss_roll(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel; α=1.0, d=8, alg=:kry_eigen, conj=false)

fig_data = plot3D_data(X, X_label);
fig_matrix_diff = plot_matrix_coords(dmap.Map, 1, X_label);

figs = [fig_matrix_diff, fig_data]
save_figs(figs)
