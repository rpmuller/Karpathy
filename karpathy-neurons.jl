# Let's see whether we can reproduce some of these using Flux.jl.
# Of course, Karpathy is doing this with torch.
using Flux

# I think this is essentially the same network that AK uses in micrograd:
model = Dense(2,1,tanh)
out = model([2,0])


# We're going to use our Value object from before to build something similar to this:
include("Value.jl")
struct Neuron
    w::Vector{Value}
    b::Value
end
function Neuron(n::Int64)
    rando() = 2*rand(Float64)-1
    w = [Value(rando(),"w$i") for i in 1:n]
    b = Value(rando(),"b")
    return Neuron(w,b)
    print(b)
end

# Call neuron object as a function:
call(neur::Neuron,x::Vector{Value}) = tanh(transpose(neur.w)*x+neur.b)
call(neur::Neuron,x::Vector{Float64}) = call(neur,[Value(x[i],"x$i") for i in 1:length(x)])
(neur::Neuron)(x::Vector{Value}) = call(neur,x)
(neur::Neuron)(x::Vector{Float64}) = call(neur,[Value(x[i],"x$i") for i in 1:length(x)])

o = Neuron(2)

# These are all the same:
o([Value(1.),Value(2.)])
o([1.,2.])
call(o,[1.,2.])