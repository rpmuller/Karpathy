
using Flux
using MLUtils

text = read("input.txt",String)
# length(text)

chars = sort(collect(Set(text)))
vocab_size = length(chars)
stoi = Dict( s => i for (i,s) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = string([chars[i] for i in l]...)

# encode("hii there")
# decode(encode("hii there"))

data = encode(text)
train_data, val_data = splitobs(data,at=0.7)

# block_size = 8 # max context length for predictions
# batch_size = 4 # independent sequences will we process in parallel?

function get_batch(split; batch_size=4, block_size=8)
	data = split == "train" ? train_data : val_data
	ix = rand(1:length(data)-block_size,batch_size)
	x = [data[ix[i]+j-1] for i in 1:batch_size, j in 1:block_size]
	y = [data[ix[i]+j] for i in 1:batch_size, j in 1:block_size]
	return x,y
end

function get_loss(X,Y) 
	Yout = model(X)
	Yoh = Flux.onehotbatch(Y,1:vocab_size)
	return Flux.logitcrossentropy(Yout,Yoh)
end

xb,yb = get_batch("train")
# xb
# yb

# Bigram language model:
model = Flux.Embedding(vocab_size => vocab_size)
ps = Flux.params(model)

# This returns a different shape than AK's bigram embedding layer.
# His returns B,T,C = 8,4,65.
# Mine return 65,8,4 (or 66 if on windows, since \n and \r are diff)
out = model(xb)
yoh = Flux.onehotbatch(yb,1:vocab_size)

out2 = permutedims(out,(3,2,1))
yoh2 = permutedims(yoh,(3,2,1))

loss1 = Flux.logitcrossentropy(out,yoh,dims=1)
loss2 = Flux.logitcrossentropy(out2,yoh2)

opt = ADAM(1e-5)

for epoch in 1:500
	x,y = get_batch("train")
	Flux.train!(get_loss,ps,[(x,y)],opt)
	println("$epoch: $(get_loss(x,y))")
end

xb,yb = get_batch("train")
xb
yb


transpose(model(xb))

transpose(xb)