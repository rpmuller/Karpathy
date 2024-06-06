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
#     display_name: hacking
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Makemore part 3
#
# https://youtu.be/P6sfmUTpUmc?si=KLAWe9fWV94wBhZp
#
# Move to more complex neural nets from MLPs.
#
# Understand MLP activation and gradients.
#
# AK asserts that RNNs are good but not particularly optimizable, and the developments since then address the optimizability.

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
# build the focabulary of characters and mappings from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# %%
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


# %%
import random
random.seed(42)
random.shuffle(words)

nwords = len(words)

n1 = int(0.8*nwords)
n2 = int(0.9*nwords)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,Yte = build_dataset(words[n2:])


# %%
# MLP revisited
n_embed = 10 # dimensionaity of the character embedding vectors
n_hidden = 200 # number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(vocab_size, n_embed,                generator=g)
W1 = torch.randn(n_embed * block_size, n_hidden,    generator=g)
b1 = torch.randn(n_hidden,                          generator=g)
W2 = torch.randn(n_hidden, vocab_size,              generator=g)
b2 = torch.randn(vocab_size,                        generator=g)


# %%
parameters = [C, W1, b1, W2, b2]
n_param = sum(p.nelement() for p in parameters)
print(n_param)
for p in parameters:
    p.requires_grad = True



# %%
# same optimization as last time:
max_steps = 200_000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

    # forward pass
    emb = C[Xb] # embed chars into vectors
    embcat = emb.view(emb.shape[0], -1) # concat the vectors
    hpreact = embcat @ W1 + b1 # hidden layer pre-activation
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb)

    # backward pass:
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100_000 else 0.01 # learning rate decay
    for p in parameters:
        p.data += -lr * p.grad
    
    # Track stats
    if i % 10_000 == 0: # print occasionally
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())


# %%
plt.plot(lossi)


# %%
@torch.no_grad()
def split_loss(split):
    x,y = {
        'train' : (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x] # (N, block_size, n_embed)
    embcat = emb.view(emb.shape[0],-1) # concat into N, block_size*n_embed
    h = torch.tanh(embcat @ W1 + b1) # (N, n_hidden)
    logits = h @ W2 + b2 # N, vocab_size
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')


# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        # forward pass nn
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)

        # sample from distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift context window and track samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))


# %%
