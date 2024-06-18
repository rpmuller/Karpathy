
using Flux
using MLUtils

text = read("input.txt",String)
# length(text)

chars = sort(collect(Set(text)))
vocab_size = length(chars)
stoi = Dict( s => i for (i,s) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = string([chars[i] for i in l]...)

encode("hii there")
decode(encode("hii there"))

data = encode(text)
train_data, val_data = splitobs(data,at=0.7)