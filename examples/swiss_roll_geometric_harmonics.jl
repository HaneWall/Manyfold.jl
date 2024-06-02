using Manyfold, Random, MLJ

rng = MersenneTwister(1)

ε_1 = 1.5
ε_2 = 0.1


kernel_dmap = GaussianKernel(ε_1)
kernel_GH = GaussianKernel(ε_2)


n = 13000
noise = 0.03
X, X_label = swiss_roll(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel_dmap; α=1.0, d=16, alg=:kry_eigen, conj=true)
# we can see by eyes for now, that the coordinates 1 and 5 are the non-harmonic ones
ψ_embedding = copy(dmap.Map[:, [1, 5]])


rng_common = 123
ψ_train, ψ_test = partition(ψ_embedding, 0.15, rng=rng_common)
X_train, X_test = partition(X', 0.15, rng=rng_common)

ψ_train_col = transpose(ψ_train)
ψ_test_col = transpose(ψ_test)


geom_all = fit(GeometricHarmonics, ψ_train_col, X_train, kernel_GH; d=61, alg=:eigen)


X_pred_all = Manyfold.predict(geom_all, ψ_test_col)
