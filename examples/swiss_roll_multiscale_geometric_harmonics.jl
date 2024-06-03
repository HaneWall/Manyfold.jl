using Manyfold, Random, MLJ

rng = MersenneTwister(1)

ε_1 = 1.5


kernel_dmap = GaussianKernel(ε_1)


n = 13000
noise = 0.02
X, X_label = swiss_roll(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel_dmap; α=1.0, d=16, alg=:kry_eigen, conj=true)
# we can see by eyes for now, that the coordinates 1 and 5 are the non-harmonic ones
ψ_embedding = copy(dmap.Map[:, [1, 5]])


rng_common = 123
ψ_train, ψ_test = partition(ψ_embedding, 0.20, rng=rng_common)
X_train, X_test = partition(X', 0.20, rng=rng_common)

ψ_train_col = transpose(ψ_train)
ψ_test_col = transpose(ψ_test)


geom_all = fit(MultiScaleGeometricHarmonics, ψ_train_col, X_train; δ=1e-5, μ=2.0)
X_pred_all = Manyfold.predict(geom_all, ψ_test_col)
