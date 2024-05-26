# Let's see whether we can reproduce some of these using Flux.jl.
# Of course, Karpathy is doing this with torch.
using Flux

# I think this is essentially the same network that AK uses in micrograd:
model = Dense(2,1,tanh)
out = model([2,0])


# We're going to use our Value object from before to build something similar to this:
include("Value.jl")

"Random number in [-1,1]"
rand1() = 2*rand()-1
Valuize(x::Vector{Float64},tag::String) = [Value(x[i],"$tag$i") for i in 1:length(x)]
Valuize(x::Vector{Float64}) = Valuize(x,"x")

struct Neuron
    w::Vector{Value}
    b::Value
end
function Neuron(n::Int64)
    w = [Value(rand1(),"w$i") for i in 1:n]
    b = Value(rand1(),"b")
    return Neuron(w,b)
    print(b)
end

# Call neuron object as a function:
call(neur::Neuron,x::Vector{Value}) = tanh(transpose(neur.w)*x+neur.b)
call(neur::Neuron,x::Vector{Float64}) = call(neur,[Value(x[i],"x$i") for i in 1:length(x)])

o = Neuron(2)
x = Valuize([2.,1.])
call(o,x)

struct Layer
    neurons::Vector{Neuron}
end
Layer(nin, nout) = Layer([Neuron(nin) for i in 1:nout])
function call(layer::Layer,x::Vector{Value}) 
    outs = [call(ni,x) for ni in layer.neurons]
    return length(outs)==1 ? outs[1] : outs
end

l = Layer(2,3)
call(l,x)

struct MLP
    layers::Vector{Layer}
end

function MLP(nin::Int64,nouts::Vector{Int64})
    sz = vcat([nin],nouts)
    layers = [Layer(sz[i],sz[i+1]) for i in 1:length(nouts)]
    MLP(layers)
end

function call(p::MLP,x::Vector{Value})
    for layer in p.layers
        x = call(layer,x)
    end
    return x
end

n = MLP(3,[4,4,1])
x = Valuize([2.,3.,-1])
call(n,x)

# at 1:51 in video. Define new inputs
xs = [2.0  3.0 -1.0;
      3.0 -1.0  0.5;
      0.5  1.0  1.0;
      1.0  1.0 -1.0]
ys = Valuize([1.0, -1.0, -1.0, 1.0])

ypred = [call(n,Valuize(xs[i,:])) for i in 1:size(xs,1)]
ys

loss = sum((ys .- ypred).^2)
loss.backward()

n.layers[1].neurons[1].w[2].grad
