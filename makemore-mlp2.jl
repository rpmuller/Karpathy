# # Karpathy Makemore using MLP
# [Video](https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s)
# Paper [A neural probabilistic language model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Bengio et al, 2003.
# The paper uses three previous words to predict a fourth word. It uses a 
# vocabulary of 17k words, implemented in a 30-dimensional space.

# This is the second part of the file, after the 
# "now made respectable" comment. Now also including changes from 
# [makemore part 3 video](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=503s)

using Flux
using StatsBase
using Plots

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
end

# Forward pass
function predict(X,C,W1,b1,W2,b2)
	emb = C[X,:]
	h = tanh.(reshape(emb, :, n_embed*block_size)*W1 .+ b1)
    return h*W2 .+ b2
end

function mloss(X,Y)
	logits = predict(X,C,W1,b1,W2,b2)
	Yoh = Flux.onehotbatch(Y,1:vocab_size)'
	# alternately
	#return Flux.logitcrossentropy(Flux.softmax(logits),Yoh,dims=2)
	return Flux.logitcrossentropy(logits,Yoh,dims=2)
end

# get_char_ix(logits) = wsample(1:27,Flux.softmax(logits[1,:])

function sample()
	out = []
	context = ones(Int64,block_size)
	while true
		logits = predict(context,C, W1, b1, W2, b2)
		ix = wsample(1:27,Flux.softmax(logits[1,:]))
		context = vcat(context[2:end],[ix])
		push!(out,ix)
		if ix == 1 break end
	end
	return string([itos[i] for i in out[1:end-1]]...)
end


# Read names.txt into words array:
words = split(read("names.txt",String),"\r\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))
vocab_size = length(itos)

# hyperparameters
block_size = 3         # context length: how many chars to we use to predict next one?
n_embed = 10        # dimension of the character embedding
n_hidden = 200     # neurons in the MLP hidden layer


n1 = 8*length(words)÷10
n2 = 9*length(words)÷10
Xtr,Ytr = build_dataset(words[1:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:end])


C = randn(vocab_size,n_embed)  # Build embedding lookup table C.

# Scaling initial values by small factor for faster optimization
scalef = 0.05
W1 = randn(n_embed * block_size, n_hidden)
b1 = randn(1,n_hidden) * scalef
W2 = randn(n_hidden,vocab_size) * scalef
b2 = randn(1,vocab_size) * scalef

ps = Flux.params(C,W1,b1,W2,b2)

learning_rate = 0.01
opt = ADAM(learning_rate)
loss_history = []

epochs = 5000
batch_size = 32 # Manually do the batching

for epoch in 1:epochs
	ix = rand(1:length(Ytr),batch_size)
    Flux.train!(mloss, ps, [(Xtr[ix,:],Ytr[ix])], opt)
    train_loss = mloss(Xtr[ix,:], Ytr[ix])
    push!(loss_history, train_loss)
    println("Epoch = $epoch: Training Loss = $train_loss")
end	

# Evaluate convergence and losses:
plot(loss_history)
mloss(Xtr,Ytr)
mloss(Xdev,Ydev)
mloss(Xte,Yte)

sample()