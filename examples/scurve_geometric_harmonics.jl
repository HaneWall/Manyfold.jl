using Manyfold, Random, MLJ

rng = MersenneTwister(1)

ε_1 = 0.1
ε_2 = 0.3

kernel_dmap = GaussianKernel(ε_1)
kernel_GH = GaussianKernel(ε_2)

n = 13000
noise = 0.03
X, X_label = scurve(n, noise; segments=n, rng=rng)

dmap = fit(DiffusionMap, X, kernel_dmap; α=1.0, d=10, alg=:kry_eigen, conj=true)
# we can see by eyes for now, that the coordinates 1 and 5 are the non-harmonic ones
ψ_embedding = copy(dmap.Map[:, [1, 5]])


rng_common = 123
ψ_train, ψ_test = partition(ψ_embedding, 0.5, rng=rng_common)
X_train, X_test = partition(X', 0.5, rng=rng_common)

#TODO fix scales when using conjugate symmetric form, fix α handling of density
geom_all = fit(GeometricHarmonics, ψ_train', X_train, kernel_GH; d=20, conj=false, α=0.0)

X_train_x = reshape(X_train[:, 1], size(X_train)[1], 1)
X_train_y = reshape(X_train[:, 2], size(X_train)[1], 1)
X_train_z = reshape(X_train[:, 3], size(X_train)[1], 1)

geom_x = fit(GeometricHarmonics, ψ_train', X_train_x, kernel_GH; conj=false, α=0.0, d=20)
geom_y = fit(GeometricHarmonics, ψ_train', X_train_y, kernel_GH; conj=false, α=0.0, d=20)
geom_z = fit(GeometricHarmonics, ψ_train', X_train_z, kernel_GH; conj=false, α=0.0, d=20)

X_pred_all = Manyfold.predict(geom_all, ψ_test')

X_pred_x = Manyfold.predict(geom_x, ψ_test')
X_pred_y = Manyfold.predict(geom_y, ψ_test')
X_pred_z = Manyfold.predict(geom_z, ψ_test')
