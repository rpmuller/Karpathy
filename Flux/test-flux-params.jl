using Flux
using Flux: gradient, params

W = randn(3, 5)
b = zeros(3)
x = rand(5)

Flux.params([W,b])
