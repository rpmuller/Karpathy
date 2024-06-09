# # Karpathy Makemore using MLP
# [Video](https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s)
# Paper [A neural probabilistic language model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Bengio et al, 2003.
# The paper uses three previous words to predict a fourth word. It uses a 
# vocabulary of 17k words, implemented in a 30-dimensional space.

using Flux
using Flux: softmax, crossentropy
using Statistics

# Read names.txt into words array:
words = split(read("names.txt",String),"\r\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))

# Compile dataset for neural net:
block_size = 3 # context length: how many chars to we use to predict next one?
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
	return X,Y' # note transpose
end

n1 = 8*length(words)÷10
n2 = 9*length(words)÷10
Xtr,Ytr = build_dataset(words[1:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:end])
Xsm,Ysm = build_dataset(words[1:100])

# 2 is a hyperparameter - the dimension of the character embedding
C = randn(27,2)  # Build embedding lookup table C.

emb = C[X,:]

W1 = randn(6,100)
b1 = randn(100)'

h = tanh.(reshape(emb,(size(emb,1),6))*W1 .+ b1)

W2 = randn(100,27)
b2 = randn(27)'

logits = h*W2 .+ b2

counts = exp.(logits)
prob = counts./sum(counts,dims=2)
sum(prob,dims=2) # Check normalization and size

prob[:,Y]

# This isn't the same
#prob2 = softmax(logits)
#sum(prob2,dims=1)

# The loss is supposed to be something like:
theloss = -mean(log.(prob[:,Y])) # negative log likelihood

# ---- now made respectable :) ----------
# Consider putting this in a separate file.
