using Manyfold, Random, MLJ

rng = MersenneTwister(1)

ε_1 = 0.3
ε_2 = 0.8

kernel_dmap = GaussianKernel(ε_1)
kernel_GH = GaussianKernel(ε_2)

n = 13000
noise = 0.02
X, X_label = scurve(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel_dmap; α=1.0, d=10, alg=:kry_eigen, conj=true)
# we can see by eyes for now, that the coordinates 1 and 5 are the non-harmonic ones
ψ_embedding = copy(dmap.Map[:, [1, 5]])


rng_common = 123
ψ_train, ψ_test = partition(ψ_embedding, 0.2, rng=rng_common)
X_train, X_test = partition(X', 0.2, rng=rng_common)

ψ_train_col = transpose(ψ_train)
ψ_test_col = transpose(ψ_test) .* [1.2, 1.0]


geom_all = fit(GeometricHarmonics, ψ_train_col, X_train, kernel_GH; d=80, α=1.0, alg=:eigen)

X_pred_all = Manyfold.predict(geom_all, ψ_test_col)

