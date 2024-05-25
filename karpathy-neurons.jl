# Let's see whether we can reproduce some of these using Flux.jl.
# Of course, Karpathy is doing this with torch.
using Flux

# I think this is essentially the same network that AK uses in micrograd:
model = Dense(2,1,tanh)

out = model([2,0])
out



# We're going to use our Value object from before to build something similar to this:
include("Value.jl")
struct Neuron
    nin::Int64
    w::Vector{Value}
    b::Value
end
function Neuron(n)
    rando() = 2*rand(Float64)-1
    w = [Value(rando(),"w$i") for i in 1:n]
    b = Value(rando(),"b")
    return Neuron(n,w,b)
    print(b)
end

function (neur::Neuron)(x::Vector{Value})
    return transpose(neur.w)*x+neur.b
end
function (neur::Neuron)(x::Vector{Float64})
    return sum([neur.w[i]*Value(x[i],"x$i") for i in 1:length(x)]) + neur.b
end

o = Neuron(2)

o([Value(1.),Value(2.)])
o.w
transpose(o.w)*[Value(1),Value(2)]
o([1.,2.])