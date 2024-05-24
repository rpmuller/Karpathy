# Learning to identify MNIST digits using Flux.jl
#  video at: https://www.youtube.com/watch?v=zmlulaxatRs

using Flux, Images, MLDatasets

using Flux: crossentropy, onecold, onehotbatch, train!, params

using LinearAlgebra, Random, Statistics

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

learning_rate = 0.05
opt = ADAM(learning_rate)

loss_history = []

epochs = 100

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end

# at 16:26 in video