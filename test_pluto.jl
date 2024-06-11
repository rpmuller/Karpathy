# Can't get this to work. Supposedly if 
# the Julia extension is installed in vs code
# you can open a Julia file in Pluto.

# There is a Pluto extension that I'm not
# going to mess with now.

a = ones(2,3)
b = ones(1,3)
a .+ b

using Flux

y_label = Flux.onehotbatch([0, 1, 2, 1, 0], 0:2)
y_model = softmax(reshape(-7:7, 3, 5) .* 1f0)
sum(y_model; dims=1)
Flux.crossentropy(y_model, y_label)
