# Karpathy MLP using Neural Nets.
# Currently at 1:04 in the vid
# Doing the same work as karpathy-makemore1.jl
# but using neural nets.
using Plots
using Flux
using Flux: onehot,onehotbatch

xs, ys = [], []
for word in words[1:1]
    chs = vcat('.',collect(word),'.')
    for (ch1,ch2) in zip(chs,chs[2:end])
        ix1,ix2 = stoi[ch1],stoi[ch2]
        push!(xs,ix1)
        push!(ys,ix2)
    end
end

# This is creating the transpose of what AK has.
# I think because of the vertical vs horiz vector
# confusion in Julia.
# Can put a transpose in here if desired:
xenc = onehotbatch(xs,1:27)'
#heatmap(xenc)

# Flux uses bools for onehot encoding.
# Doesn't seem like I have to worry about it:
W = randn(27,27)
logits = xenc*W # logits = log-counts:

# Need to get outputs as probabilities, which should be 
# positive and sum to one. Interpret these outputs as 
# log-counts. We want to exponentiate and then normalize
# these to get probs. This is called softmax.

probs = exp.(logits)
for i in 1:size(probs)[1]
    probs[i,:] /= sum(probs[i,:])
end


# at 1:29 looking in detail at one example .emma.
nlls = zeros(Float64,5)
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
println("========")
println("average neg log likelihood, i.e. loss: $(mean(nll))")

# He's now doing manual optimization of the simple linear net.
# I'm not going to do all this.

# There's something nice about how torch has both the ability to 
# create arrays and to apply networks and functions. I like the 
# fact that Flux is cleaner and just uses Julia arrays, but I 
# haven't found the analogous capability in Flux.

# The problem with this example is that we don't really know
# when we're doing a good job, unlike MNIST or linear regression.
