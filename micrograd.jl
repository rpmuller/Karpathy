using Plots

# # Karpathy NN Zero to Hero
#
# I'm going to attempt to redo (some of?) Karpathy's 
# [Neural Nets: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
# tutorials in Julia. 
#
# Why rewrite? I've worked through much of this in Python, and seeing how things 
# are different in another language, perhaps without the crutch of using pytorch, 
# will make me learn the theory more deeply.
#
# Why Julia? It's fast, elegant, and just as readable as Python. (Although switching 
# back and forth between the two often leaves me forgetting which language I'm using 
# and how to do the most basic things.)
#
# There's a lot of work going on to rewrite everything in C to have fast code and simple 
# access to GPUs. A lot of this could just as easily be done in Julia and produce easier 
# to understand code that will (hopefully) be just as fast.


# # Micrograd
# Starting with the [Micrograd](https://github.com/karpathy/micrograd) implementation.

# [Micrograd Youtube Video](https://www.youtube.com/watch?v=VMj-3S1tku0).

# Understanding derivatives, creating functions, plotting them.

# Julia has a nice syntax for creating functions in one line.
f(x) = 3*x^2 -4*x + 5
f(3)
f(3.0)
# These are (intentionally) type vague for now, so calling with an int returns an int, 
# and calling with a float returns a float.

# Compute (numerical) derivatives of function
h = 0.00001
x = 2/3
(f(x+h)-f(x))/h

include("Value.jl")

plot(-5:0.25:5,f, title="Plotting functions in Julia", label="f(x)")


a2 = Value(2.0,"a2")
b2 = Value(-3.0,"b2")
c2 = Value(10,"c2")

a2*b2
d2 = a2*b2+c2
d2.prev

using Graphs
using GraphPlot

## Graph visualization package for Values
# This starts in AK's video around 25:44.
# Using Julia's [Graphs.jl](https://juliagraphs.org/Graphs.jl/dev/) package.

"Build a set of all nodes and edges in a graph"
function trace(root)
	nodes = Set()
	edges = Set()
	function build(v)
		if v ∉ nodes
			push!(nodes,v)
			for child in v.prev
				push!(edges,(child,v))
				build(child)
			end #for
		end #if
	end # function
	build(root)
	return nodes, edges
end

"Use `Graph` and `GraphPlot` to draw a graph with labeled nodes"
function draw(root)
	nodes,edges = trace(root)
	nodelist = collect(nodes)

	g = DiGraph(length(nodes))

	for (child,parent) in edges
		ich = findfirst(==(child),nodelist)
		ipar = findfirst(==(parent),nodelist)
		add_edge!(g,ich,ipar)
	end
	nodelabels = ["$(node.label)=$(round(node.data,digits=3))" for node in nodelist]
	gplot(g,nodelabel=nodelabels)
end

draw(d2)

## Back Propagation"
# Neural nets use the `tanh` activation function to squash response values to [-1,1].
plot(-5:0.1:5,tanh,label="tanh")

# Manually create a neural net
# Inputs:
x1 = Value(2.0,"x1")
x2 = Value(0.0,"x2")

# Weights:
w1 = Value(-3.0,"w1")
w2 = Value(1.0,"w2")

# Bias:
b = Value(6.88137358,"b")

# Create network:
x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1+x2w2
n = x1w1x2w2 + b
o = tanh(n)

draw(o)

o.grad

backprop(o)
b.grad

x2.grad
x1.grad



# Expand capabilities of Value type to support broader array of math
Value(1) # integer
a2 + 1
1 + a2
a2 * 2
2 * a2

exp(a2)

a2^2
a2/b2
a2 * (1/b2)
a2 * (b2^(-1))
a2 / 2
