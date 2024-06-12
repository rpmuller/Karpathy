# # Karpathy Makemore using MLP
# [Video](https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s)
# Paper [A neural probabilistic language model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Bengio et al, 2003.
# The paper uses three previous words to predict a fourth word. It uses a 
# vocabulary of 17k words, implemented in a 30-dimensional space.

# This is the second part of the file, after the 
# "now made respectable" comment

using Flux
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

function mloss(X,Y)
	logits = predict(X,C,W1,b1,W2,b2)
	Yoh = Flux.onehotbatch(Y,1:27)'
	loss = Flux.logitcrossentropy(logits,Yoh,dims=2)
	# alternately
	#loss = Flux.logitcrossentropy(Flux.softmax(logits),Yoh,dims=2)
	return loss
end

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

emb = C[Xsm,:]

ps = Flux.params(C,W1,b1,W2,b2)

learning_rate = 0.01
opt = ADAM(learning_rate)

loss_history = []

epochs = 50

Xin,Yin = Xtr,Ytr
# Manually do the batching
bsize = 32

for epoch in 1:epochs
	ix = rand(1:length(Yin),bsize)
    Flux.train!(mloss, ps, [(Xin[ix,:],Yin[ix])], opt)
    train_loss = mloss(Xin, Yin)
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end	
