# Learning to identify MNIST digits using Flux.jl
#  video at: https://www.youtube.com/watch?v=zmlulaxatRs

using Flux, Images, MLDatasets
using Flux: crossentropy, onecold, onehotbatch, train!, params
using LinearAlgebra, Random, Statistics
using Plots

# set random seed

Random.seed!(3145926)

X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]

# view training data
idx = 7
img = X_train_raw[:,:,idx]
colorview(Gray, img') #' is the transpose
y_train_raw[idx]

# Flatten data
X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

# one-hot encoding
y_train = onehotbatch(y_train_raw, 0:9)
y_test = onehotbatch(y_test_raw,0:9)

model = Chain(
    Dense(28*28, 32, relu),
    Dense(32, 10),
    softmax
)

loss(x,y) = crossentropy(model(x), y)

ps = params(model) # initialize all parameters

learning_rate = 0.01
opt = ADAM(learning_rate)

loss_history = []

epochs = 500

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end

# make predictions
y_hat_raw = model(X_test)
y_hat = onecold(y_hat_raw) .- 1
y = y_test_raw
mean(y_hat .== y) # doggo got 96.24%
# CNN gets 99.83% according to wikipedia

# display results
check = [y_hat[i] == y[i] for i in 1:length(y)]
index = collect(1:length(y))
check_display = [index y_hat y check]
vscodedisplay(check_display)

misclass_index=9

img = X_test_raw[:,:,misclass_index]

colorview(Gray, img')
y[misclass_index]
y_hat[misclass_index]

gr(size = (600,600))
pl = plot(1:epochs, loss_history, xlabel="Epochs", ylabel="Loss", title="Learning Curve", legend=false,
    color=:blue, linewidth=2)