# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Makemore using a Multilayer Perceptron
#
# Based on [A Neural Probabilitic Language Model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by Bengio et al.
#
# They're using a word-based network, whereas we're still using a character based network.

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
# read in all the words
words = open('names.txt','r').read().splitlines()
words[:8]

# %%
len(words)

# %%
# build the focabulary of characters and mappings from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# %%
# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one:
X, Y = [], []

for w in words:
    #print(w)
    context = [0]*block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

# %%
X.shape, X.dtype,Y.shape, Y.dtype

# %%
C = torch.randn((27,2)) # map all 27 characters to a 2 dim space

# %%
# we can use pytorch indexing to embed the vectors.
# We used one_hot encoding before, but we're just using
# indexing now, for simplicity and to skip a matmult
emb = C[X]
emb.shape

# %%
# randomize weights and biases.
W1 = torch.randn((6,100)) # 6 = 3*2, from the last two indices of C
b1 = torch.randn(100)

# %%
# concatenate embeddings across dimension 1 (blocksize)
torch.cat([emb[:,0,:],emb[:,1,:],emb[:,2,:]])

# %%
# Torch.unbind generalizes this if we want to change the blocksize
torch.cat(torch.unbind(emb,1),1)

# %%
# There is a better way to do this: view is much faster in torch
emb.view(32,6)

# %%
# we can now do the proper math for the nn:
#h = emb.view(32,6) @ W1 + b1
#h = emb.view(emb[0],6) @ W1 + b1
h = torch.tanh(emb.view(-1,6) @ W1 + b1)

# %%
h.shape

# %%
# Mapping back to 27 characters
W2 = torch.randn((100,27))
b2 = torch.randn(27)

# %%
logits = h @ W2 + b2

# %%
logits.shape

# %%
counts = logits.exp()

# %%
prob = counts / counts.sum(1,keepdims=True)

# %%
prob.shape

# %%
# actual error comes from Y which is the prediction
loss = -prob[torch.arange(32), Y].log().mean()

# %%
loss

# %%
# rewrite everything and make more respectable
X.shape, Y.shape # dataset

# %% [markdown]
# I wish he would give the numbers 2, 100, 6 names
#
# - 32 is the number of training items from the first 5 words
# - 27 is nchar
# - 2 is the size of an intermediate layer
# - 3 is the context width
# - 6 is 2*3

# %%
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27,generator=g)
parameters = [C, W1, b1, W2, b2]

# %%
sum(p.nelement() for p in parameters) # number of parameters in total

# %%
for p in parameters:
    p.requires_grad = True

# %%
niter = 1000
for _ in range(niter):

    # minibatch construct
    # we're no longer using the true gradient, but it's still good enough
    # it's much better to use an approximate gradient and take more steps
    #  than it is to use the exact gradient and take fewer steps.
    ix = torch.randint(0, X.shape[0], (32,))

    # forward pass
    emb = C[X[ix]] # 32,3,2
    h = torch.tanh(emb.view(-1,6)@W1 + b1) # 32,100
    logits = h@W2 + b2 # 32,27
    loss = F.cross_entropy(logits,Y[ix]) # use this for efficiency and to prevent overflow

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data += -0.1 * p.grad

print(loss.item())


# %%
# evaluate full loss
emb = C[X]
h = torch.tanh(emb.view(-1,6)@W1 + b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits,Y)
loss

# %%
# discussion on different learning rates, how 0.1 is a good choice
#  and something about how you might degrade the lerning rate after
#  things begin to converge.


# %%
# discussion about overfitting.
#  split dataset into training, dev/validate, and test
#  80%, 10%, 10%
#  dev is used to tune number of hyperparameters
#  test is used to evaluate model at end.
