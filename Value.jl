# # Values
# Create a new struct in Julia called `Value` analogous to what AK did in Python. 
# Overload algebraic functions. Keep track of children so we can backprop.

import Base

"Null function takes no arguments and returns nothing"
nullf() = nothing
mutable struct Value
    data::Float64
    grad::Float64
    prev::Vector{Value}
    label::String
    backward
end
Value(d) = Value(d,0,[],"",nullf) # Outer constructor
Value(d,l) = Value(d,0,[],l,nullf) # data plus label

function Base.show(io::IO, v::Value)
    print(io,"Value($(v.label)=$(round(v.data,digits=3)))")
end

Base.transpose(v::Value) = v

function Base.:(+)(v1::Value, v2::Value)
    newlabel = "$(v1.label)+$(v2.label)"
    out = Value(v1.data+v2.data,0,[v1,v2],newlabel,nullf)
    
    function back()
        v1.grad += 1.0 * out.grad
        v2.grad += 1.0 * out.grad
        return nothing
    end
    out.backward = back
    return out 
end
Base.:(+)(v1::Value, v2::Number) = v1 + Value(v2,"n")
Base.:(+)(v2::Number, v1::Value) = v1 + Value(v2,"n")

function Base.:(*)(v1::Value, v2::Value)
    newlabel = "$(v1.label)*$(v2.label)"
    out = Value(v1.data*v2.data,0,[v1,v2],newlabel,nullf)

    function back()
        v1.grad += v2.data * out.grad
        v2.grad += v1.data * out.grad
        return nothing
    end
    out.backward = back
    return out
end
Base.:(*)(v1::Value, v2::Number) = v1 * Value(v2,"n")
Base.:(*)(v2::Number, v1::Value) = v1 * Value(v2,"n")

function Base.tanh(v::Value)
    x = 2v.data
    t = (exp(x)-1)/(exp(x)+1)
    newlabel = "tanh($(v.label))"
    out = Value(t,0,[v],newlabel,nullf)

    function back()
        v.grad += (1-t^2)*out.grad
        return nothing
    end
    out.backward = back
    return out
end

function Base.exp(v::Value)
    x = v.data
    t = exp(x)
    newlabel = "exp($(v.label))"
    out = Value(t,0,[v],newlabel,nullf)

    function back()
        v.grad += t*out.grad
        return nothing
    end
    out.backward = back
    return out
end

function Base.:(^)(v::Value, k::Number)
    x = v.data
    t = x^k
    newlabel = "$(v.label)^$k"
    out = Value(t,0,[v],newlabel,nullf)

    function back()
        v.grad += k*x^(k-1)*out.grad
        return nothing
    end
    out.backward = back
    return out
end

function Base.inv(v::Value)
    x = v.data
    t = x^(-1)
    newlabel = "inv($(v.label))"
    out = Value(t,0,[v],newlabel,nullf)

    function back()
        v.grad += -x^(-2)*out.grad
        return nothing
    end
    out.backward = back
    return out
end


Base.:(/)(v1::Value, v2::Value) = v1*inv(v2)
Base.:(/)(v1::Value, n::Number) = v1*inv(Value(n,"n"))
Base.:(/)(n::Number, v::Value) = Value(n,"n")*inv(v)


function backprop(root::Value)
    topo = []
    visited = Set()
    function build_topo(v)
        if v ∉ visited
            push!(visited,v)
            for child in v.prev
                build_topo(child)
            end
            push!(topo,v)
        end
    end
    build_topo(root)
    root.grad = 1
    for node in reverse(topo)
        node.backward()
    end
    return nothing
end
