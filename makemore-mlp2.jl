# # Karpathy Makemore using MLP
# [Video](https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s)
# Paper [A neural probabilistic language model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Bengio et al, 2003.
# The paper uses three previous words to predict a fourth word. It uses a 
# vocabulary of 17k words, implemented in a 30-dimensional space.

# This is the second part of the file, after the 
# "now made respectable" comment

using Flux
using Flux: train!, params, gradient, crossentropy, softmax, DataLoader, update!
using Statistics

# hyperparameters
embedding_depth = 2    # dimension of the character embedding
block_size = 3         # context length: how many chars to we use to predict next one?

# Read names.txt into words array:
words = split(read("names.txt",String),"\r\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))

# Compile dataset for neural net:
function build_dataset(words)
	X0 = []
	Y::Array{Int64} = []
	for w in words
		context = ones(Int64,block_size)
		for ch in string(w,".")
			ix = stoi[ch]
			push!(X0,context)
			push!(Y,ix)
			context = vcat(context[2:end],[ix])
		end
	end
	nrows = length(X0)
	ncols = length(X0[1])
	X = zeros(Int64,nrows,ncols)
	for i in 1:nrows
    	X[i,:] = X0[i]
	end
	return X,Vector(Y) # note transpose
	#return X,Y' # note transpose
end

# Forward pass
function predict(X,C,W1,b1,W2,b2)
	emb = C[X,:]
	h = tanh.(reshape(emb,(size(emb,1),6))*W1 .+ b1)
    return h*W2 .+ b2
end

# TODO: get cross entropy working below. Currently Y is the wrong shape.
function custom_crossentropy(logits,Y)
    # These now match:
    #prob = counts./sum(counts,dims=1)
    prob = softmax(logits)
	loss = -mean(log.([prob[i,Y[i]] for i in 1:length(Y)])) # negative log likelihood
	#loss = crossentropy(logits,Y) # Doesn't give the same result
    return loss
end

mloss(X,Y) = custom_crossentropy(predict(X,C,W1,b1,W2,b2),Y)

n1 = 8*length(words)÷10
n2 = 9*length(words)÷10
Xtr,Ytr = build_dataset(words[1:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:end])
Xsm,Ysm = build_dataset(words[1:100])

C = randn(27,embedding_depth)  # Build embedding lookup table C.

W1 = randn(6,100)
b1 = randn(1,100)

W2 = randn(100,27)
b2 = randn(1,27)

logits = predict(Xsm,C,W1,b1,W2,b2)
#counts = exp.(logits)
prob = counts./sum(counts,dims=1)
prob2 = softmax(logits)
loss = -mean(log.([prob[i,Ysm[i]] for i in 1:length(Ysm)])) # negative log likelihood
loss2 = crossentropy(logits,Ysm) # Doesn't give the same result


emb = C[Xsm,:]
emb2 = reshape(emb,size(emb,1),6)
emb2*W1.+b1

ps = Flux.params(C,W1,b1,W2,b2)

learning_rate = 0.01
opt = ADAM(learning_rate)

loss_history = []

epochs = 50

# TODO: get code working with minibatch. Currently works with Xsm, but runs 
# out of memory for larger inputs.
Xin,Yin = Xsm,Ysm
size(Xin)
size(Yin)
length(Yin)
data = [(Xin,Yin)] # no minibatches
#data = DataLoader((Xin,Yin),batchsize=50,partial=false)
for epoch in 1:epochs
    #train!(mloss, ps, data, opt)
    grads = gradient(()->mloss(Xin,Yin),ps)
    for p in ps
        update!(p,-learning_rate * grads[p])
    end
    train_loss = mloss(Xin, Yin)
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end	

f[1 2 3; 4 5 6] .+ vec([1,1])

ones(2,1)