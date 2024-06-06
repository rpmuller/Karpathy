# Karpathy's Neural Net Zero-to-Hero experiments

I'm going to attempt to redo (some of?) Karpathy's 
[Neural Nets: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) 
tutorials in Julia. 

Why rewrite? I've worked through much of this in Python, and seeing how things are different in another language, perhaps without the crutch of using pytorch, will make me learn the theory more deeply.

Why Julia? It's fast, elegant, and just as readable as Python. (Although switching back and forth between the two often leaves me forgetting which language I'm using and how to do the most basic things.)

There's a lot of work going on to rewrite everything in C to have fast code and simple access to GPUs. A lot of this could just as easily be done in Julia and produce easier to understand code that will (hopefully) be just as fast.

Tasks
- Finish cleaning directories: make sure all jupyter notebooks are converted to julia and then delete, since they don't sync well.
