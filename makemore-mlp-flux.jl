# Redo makemore-mlp2 but using more standard Flux
# Chains of built-in layers.

using StatsBase
using Flux

function build_dataset(words)
	X = []
	Y::Array{Int64} = []
	for w in words
		context = ones(Int64,block_size)
		for ch in string(w,".")
			ix = stoi[ch]
			push!(X,context)
			push!(Y,ix)
			context = vcat(context[2:end],[ix])
		end
	end
	return vcat(transpose.(X)...),vec(Y)
end

get_char_ix(logits) = wsample(1:27,softmax(logits))
get_char(logits) = chars[get_char_ix(logits)]

stringify(ixs) = string([chars[ix] for ix in ixs[1:end-1]]...)

function get_name(model)
    out = []
    context = ones(Int64,block_size)
    while true
        logits = model(reshape(context,block_size,1))
        ix = get_char_ix(vec(logits))
        context = vcat(context[2:end],[ix])
        push!(out,ix)
        if ix == 1 break end
    end
    return stringify(out)
end

loss(X,Yoh) = Flux.logitcrossentropy(model(X'),Yoh)

# Read names.txt into words array:
words = split(read("names.txt",String),"\r\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
vocab_size = length(chars)

block_size = 3

n1 = 8*length(words)÷10
n2 = 9*length(words)÷10
Xtr,Ytr = build_dataset(words[1:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:end])
Xsm,Ysm = build_dataset(words[1:100])

Xsm[9,:]


Xsm
# Build a model
n_embedding = 10
n_hidden = 100
model = Chain(
    Flux.Embedding(vocab_size => n_embedding),
	Flux.flatten,

	# Not certain BatchNorm is helping things, but maybe I'm using it wrong.
    #Dense(block_size*n_embedding => n_hidden,bias=false), 
	#BatchNorm(n_hidden,tanh),
    Dense(block_size*n_embedding => n_hidden,tanh),
    
	Dense(n_hidden => vocab_size)
)
ps = Flux.params(model)

model[3]

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
