
# Karpathy Makemore using Neural Nets
# This is the second part of the 
# [First Makemore Lesson](https://www.youtube.com/watch?v=PaCmpygFfXo) which starts around 1 hr in. AK has already built a bigram model for the names app, 
# and is now extending it using torch arrays and autodiff.

using Plots
using Flux
using Flux: onehotbatch
using Statistics, StatsBase

# Read names.txt into words:
f = open("names.txt","r")
    s = read(f,String)
close(f)
words = split(s,"\n")
words[1:10]

# Make character mapping arrays
chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = Dict( s => i for (i,s) in enumerate(chars))
itos = Dict( i => s for (i,s) in enumerate(chars))

# Create the count array and fill it
N = zeros(Int32,27,27)
for word in words
    chs = vcat('.',collect(word),'.')
    for (ch1,ch2) in zip(chs,chs[2:end])
        ix1,ix2 = stoi[ch1],stoi[ch2]
        N[ix1,ix2] += 1
    end
end

# Might be worth some time getting his plotting to work.
# But not now.
heatmap(N)

# Example of multinomial sampling that AK goes through at 29:00:
#histogram(sample(0:2,pweights([0.6064, 0.3033,0.0903]),10000))

# Create a matrix of the letter probabilities
nrows,ncols = size(P)
P = float(N.+1) # The 1 here is the model smoothing
for i in 1:nrows
    P[i,:] /= sum(P[i,:])
end

# Sample the distribution of letters:
for _ in 1:20
    ix = 1
    name = []
    while true
        row = N[ix,:]
        p = P[ix,:]
        ix = sample(1:27,pweights(p),1)[1]
        if (ix==1) break end
        push!(name,itos[ix])
    end
    println(join(name))
end

# Evaluate the quality of the bigram model we've constructed.
# Use avg neg log likelihood
log_likelihood = 0
n = 0
for word in words
    chs = vcat('.',collect(word),'.')
    for (ch1,ch2) in zip(chs,chs[2:end])
        ix1,ix2 = stoi[ch1],stoi[ch2]
        log_likelihood += log(P[ix1,ix2])
        n += 1
    end
end    
nll = -log_likelihood
nll/n # average negative log likelihood, which is what we'll use as the loss
# we will also add a smoothing to all numbers to soften the low probs

# Since the next part of the video uses neural nets to do the same
# thing, I'm going to move this into another file.
# Currently at 1:04, and I'm moving to karpathy-makemore1-nn.jl
