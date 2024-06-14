using Flux

# Read names.txt into words array:
words = split(read("names.txt",String),"\n")

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

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))
vocab_size = length(itos)

block_size = 3         # context length: how many chars to we use to predict next one?
n_embed = 10        # dimension of the character embedding
n_hidden = 200     # neurons in the MLP hidden layer
