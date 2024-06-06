# # Karpathy Makemore using MLP
# [Video](https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s)
# Paper [A neural probabilistic language model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Bengio et al, 2003.
# The paper uses three previous words to predict a fourth word. It uses a 
# vocabulary of 17k words, implemented in a 30-dimensional space.

using Plots
using Flux
using Flux: softmax, crossentropy
using Statistics

# Read names.txt into words array:
f = open("names.txt","r")
    s = read(f,String)
close(f)
words = split(s,"\r\n")

# Create character embeddings.
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))

# We're going to follow what Karpathy does pretty closely. The section of 
# the FluxML docs called 
# [Building Simple Models](https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Simple-Models) 
# gives a good foundation for doing this.

# Compile dataset for neural net:
block_size = 3 # context length: how many chars to we use to predict next one?
X0,Y = [],[]

for w in words[1:5]
    println(w)
    context = ones(Int64,block_size)
    for ch in string(w,".")
        ix = stoi[ch]
        push!(X0,context)
        push!(Y,ix)
        println(join(itos[i] for i in context)," ---> ", itos[ix])
        context = vcat(context[2:end],[ix])
    end
end

# Repack X0 matrix
nrows = length(X0)
ncols = length(X0[1])
X = zeros(Int64,nrows,ncols)
for i in 1:nrows
    X[i,:] = X0[i]
end

C = randn(27,2)  # Build embedding lookup table C.
W1 = randn(6,100)
b1 = randn(100)'
W2 = randn(100,27)
b2 = randn(27)';
params = Flux.params(C,W1,b1,W2,b2)

# Forward pass
function predict(X)
    Xemb = hcat(C[X[:,1],:],C[X[:,2],:],C[X[:,3],:]) # Build the embedded input matrix:
    h = tanh.(Xemb*W1 .+ b1)
    return h*W2 .+ b2
end

function mloss(X,Y)
    logits = predict(X)
    # This should be equivalent to softmax, but Flux's softmax doesn't give the same result:
    counts = exp.(logits)
    prob = zeros(Float64,size(counts))
    for i in 1:size(prob)[1]
        prob[i,:] = counts[i,:]/sum(counts[i,:])
    end
    #prob[1:32,Y] # This is what AK does in Python

    # This should be the same as crossentropy, but Flux's crossentropy doesn't give the same result
    results = [prob[i,Y[i]] for i in 1:32]  # Here's what does that operation in Julia
    return -mean(log.(results))
end

gs = gradient((X,Y) -> mloss(X,Y),Flux.params(C,W1,b1,W2,b2))
