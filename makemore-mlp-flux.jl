# Redo makemore-mlp2 but using more standard Flux
# Chains of built-in layers.

using StatsBase
using Flux

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
	return X,vec(Y) # note transpose
end
block_size = 3

get_char_ix(logits) = wsample(1:27,softmax(logits))
get_char(logits) = chars[get_char_ix(logits)]

get_char_ix(logits)
get_char(logits)
stringify(ixs) = string([itos[ix] for ix in ixs[1:end-1]]...)

function get_name(model)
    out = []
    context = ones(Int64,block_size)
    while true
        logits = model(reshape(context,1,3))
        ix = get_char_ix(logits)
        context = vcat(context[2:end],[ix])
        push!(out,ix)
        if ix == 1 break end
    end
    return stringify(out)
end

get_name(model)
loss(X,Yoh) = Flux.logitcrossentropy(model(X'),Yoh)

# Read names.txt into words array:
words = split(read("names.txt",String),"\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))
vocab_size = length(itos)

n1 = 8*length(words)÷10
n2 = 9*length(words)÷10
Xtr,Ytr = build_dataset(words[1:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:end])
Xsm,Ysm = build_dataset(words[1:100])

# Build a model
# I'm not really sure I'm using an embeddingbag correctly
# here. In the previous MLP study, I had 3 encoders that took 
# 27 chars to a space of 2-10. Here the eb encoder takes any 
# number of char inputs and puts them into a 30-d space.

model = Chain(
    Flux.Embedding(27 => 10),
	Flux.flatten,
    Dense(30 => 100, tanh),
    Dense(100 => 27)
)
ps = Flux.params(model)

#emb = Embedding(27 => 10)
#Flux.flatten(emb(Xsm'))

# Optimize the model
opt = ADAM(0.01)
batch_size = 50

Xin,Yin = Xtr,Ytr
Yoh = Flux.onehotbatch(Yin,1:27)
for epoch in 1:50
    ix = rand(1:length(Yin),batch_size)
    Flux.train!(loss,ps,[(Xin[ix,:],Yoh[:,ix])],opt)
    train_loss = loss(Xin,Yoh)
    println("$epoch: $train_loss")
end

# Demonstrate the model
get_name(model)
