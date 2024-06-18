
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

block_size = 8 # max context length for predictions
batch_size = 4 # independent sequences will we process in parallel?

function get_batch(split)
	data = split == "train" ? train_data : val_data
	ix = rand(1:length(data)-block_size,batch_size)
	x = [data[ix[i]+j-1] for i in 1:batch_size, j in 1:block_size]
	y = [data[ix[i]+j] for i in 1:batch_size, j in 1:block_size]
	return x,y
end

xb,yb = get_batch("train")
xb
yb



