
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

ψ_test_small = ψ_test'
preim_ψ_test = preim_knn(dmap, ψ_test_small; k=6, embed_dim=[1, 5])
