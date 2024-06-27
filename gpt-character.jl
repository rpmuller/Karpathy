
using Flux
using MLUtils
using Statistics, StatsBase

# hyperparameters
batch_size = 32 # independent sequences will we process in parallel?
block_size = 8 # max context length for predictions
epochs = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embed = 32

text = read("input.txt",String)
length(text)

chars = sort(collect(Set(text)))
vocab_size = length(chars)
stoi = Dict( s => i for (i,s) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = string([chars[i] for i in l]...)

# encode("hii there")
# decode(encode("hii there"))

data = encode(text)
train_data, val_data = splitobs(data,at=0.9)

function get_batch(split) #; batch_size=4, block_size=8)
	data = split == "train" ? train_data : val_data
	ix = rand(1:length(data)-block_size,batch_size)
	x = [data[ix[i]+j-1] for i in 1:batch_size, j in 1:block_size]
	y = [data[ix[i]+j] for i in 1:batch_size, j in 1:block_size]
	return x,y
end

function get_loss(X,Y) 
	logits = model(X)
	Yoh = Flux.onehotbatch(Y,1:vocab_size)
	return Flux.logitcrossentropy(logits,Yoh)
end

function estimate_loss()
	avgloss = Dict()
	testmode!(model)
	for split in ["train","val"]
		losses = []
		for _ in 1:eval_iters
			X,Y = get_batch(split)
			loss = get_loss(X,Y)
			push!(losses,get_loss(X,Y))
		end
		avgloss[split] = mean(losses)
	end
	trainmode!(model)
	return avgloss
end

get_char_ix(logits) = wsample(1:vocab_size,softmax(logits))
get_char(logits) = chars[get_char_ix(logits)]
stringify(ixs) = string([chars[ix] for ix in ixs[1:end-1]]...)

function generate(max_new_tokens=100)
	input = [2,1]
	for _ in 1:max_new_tokens
		logits = model(input)
		ix = get_char_ix(logits[:,end])
		push!(input,ix)
	end
	return stringify(input)
end


# Bigram language model:
model = Chain(
	Embedding(vocab_size => n_embed),
	Dense(n_embed => vocab_size)
)
ps = Flux.params(model)

opt = ADAM(1e-3)

for epoch in 1:epochs
	x,y = get_batch("train")
	Flux.train!(get_loss,ps,[(x,y)],opt)
	if epoch % eval_interval == 0 
		losses = estimate_loss()
		tloss = losses["train"]
		vloss = losses["val"]
		println("$epoch $tloss $vloss")
	end
end


out = generate(500)
println(out)


# Mathematical trick in self attention
B,T,C = 4,8,2
x = randn(B,T,C)
xbow = zeros(B,T,C)
for b in 1:B
	for t in 1:T
		xprev = x[b,1:t,:]
		xbow[b,t,:] = mean(xprev,dims=1)
	end
end
xbow

# Can use a lower triangular matrix
TriAvg(nr,nc) = [i >= j ? 1/i : 0 for i in 1:nr, j in 1:nc]

W = TriAvg(T,T)
xbow2 = zeros(B,T,C)
for b in 1:B
	xbow2[b,:,:] = W*x[b,:,:]
end
xbow2
W*x[2,:,:]

xbow ≈ xbow2

# Can also use softmax
