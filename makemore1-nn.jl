# Karpathy Makemore using Neural Nets
# This is the second part of the 
# [First Makemore Lesson](https://www.youtube.com/watch?v=PaCmpygFfXo) which starts around 1 hr in. AK has already built a bigram model for the names app, 
# and is now extending it using torch arrays and autodiff.

using Plots
using Flux
using Flux: onehotbatch
using Statistics

# Read in the names
f = open("names.txt","r")
    s = read(f,String)
close(f)
words = split(s,"\n")

# Make char mappings
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))

# Parse the names into a neural net style of inputs and outputs.
xs, ys = [], []
for word in words[1:1]
    chs = vcat('.',collect(word),'.')
    for (ch1,ch2) in zip(chs,chs[2:end])
        ix1,ix2 = stoi[ch1],stoi[ch2]
        push!(xs,ix1)
        push!(ys,ix2)
    end
end

# Use `onehot` encoding to encode the inputs. The transpose 
# is needed below because 
# of the standard Julia vert vs horiz vector confusion.
xenc = onehotbatch(xs,1:27)'

# heatmap(xenc)

# Create a random set of weights and multiply by the encoded inputs.

# Flux uses bools for onehot encoding.
# Doesn't seem like I have to worry about it:
W = randn(27,27)
logits = xenc*W # logits = log-counts

# Need to get outputs as probabilities, which should be 
# positive and sum to one. Interpret these outputs as 
# log-counts. We want to exponentiate and then normalize
# these to get probs. This is called softmax.

probs = exp.(logits)
for i in 1:size(probs)[1]
    probs[i,:] /= sum(probs[i,:])
end

nlls = zeros(Float64,5)
nll = 0
for i in 1:5
    x = xs[i]
    y = ys[i]
    println("--------")
    println("bigram example $i: $(itos[i]) $(itos[y]) indices $x, $y")
    println("input to the neural net: $x")
    println("output probabilities from neural net: $probs")
    println("label of actual next character: $y")
    p = probs[i,y]
    println("probability assigned by the net to the next char: $p")
    logp = log(p)
    println("log likelihood $logp")
    nll = -logp
    println("negative log likelihood: $nll")
    nlls[i] = nll
end

# At 1:29 in video. Looking in detail at one example .emma.

println("========")
println("average neg log likelihood, i.e. loss: $(mean(nll))")