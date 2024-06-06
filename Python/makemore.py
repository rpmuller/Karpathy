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
# # Makemore tutorial by Andrej Karpathy
#
# - [Karpathy makemore video](https://www.youtube.com/watch?v=PaCmpygFfXo)
# - 
# Makes more of things that you give it. Give it a list of names and it will create more names.
#
# Character level language model.

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# %%
words = open('names.txt','r').read().splitlines()

# %%
words[:10]

# %%
N = torch.zeros((27,27), dtype=torch.int32) # bigram letter counts

# %%
chars = 'abcdefghijklmnopqrstuvwxyz'
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# %%
for w in words:
  chrs = ["."] + list(w) + ["."] # add special start/end characters
  for ch1,ch2 in zip(chrs,chrs[1:]):
    ix1,ix2 = stoi[ch1],stoi[ch2]
    N[ix1,ix2] += 1

# %%
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
  for j in range(27):
    chstr = itos[i] + itos[j]
    plt.text(j,i,chstr, ha="center",va="bottom", color='gray')
    plt.text(j,i, N[i,j].item(), ha="center", va="top", color="gray")
plt.axis('off')

# %%
g = torch.Generator().manual_seed(2147483647)

# %%
P = N.float()
P /= P.sum(1,keepdim=True)

# %%
for i in range(10):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

# %% [markdown]
# AK covered using smoothed average negative log likelihood as a loss function. Observes a loss of 3.5 across the whole training set.

# %% [markdown]
# # Neutral Net based training
#
# Now use neutral nets to create a bigram model for words.

# %%
# create the training set of all bigrams
xs, ys = [], [] # inputs , outputs. Outputs can also be interpreted as labels for input data

for w in words[:1]:
  chrs = ["."] + list(w) + ["."] # add special start/end characters
  for ch1,ch2 in zip(chrs,chrs[1:]):
    ix1,ix2 = stoi[ch1],stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# %%
# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs.
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g)

# %%
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdim=True) # probs of next character
# last 2 lines are called a 'softmax'

# %%
nlls = torch.zeros(5)
for i in range(5):
    # ith bigram:
    x = xs[i].item() # input char index
    y = ys[i].item() # label character index
    print ('-------------------')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x}, {y})')
    print('input to the neural net: ', x)
    print('output probs from the neural net:', probs[i])
    print('label (actual next character):',y)
    p = probs[i, y]
    print('prob assigned by the net to the correct char',p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll
print('========')
print('avg negative log likelihood (loss) = ',nlls.mean().item())

# %%
# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs.
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

# %%
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdim=True) # probs of next character

# Implement a back propagation
loss = -probs[torch.arange(5), ys].log().mean()
loss.item()

# %%
# backwards pass:
W.grad = None # set grad to zero
loss.backward()

# %%
# update
W.data += -0.1*W.grad

# %%
# Put everything together from scratch:

# create the dataset
xs, ys = [], []
for w in words:
  chrs = ["."] + list(w) + ["."] # add special start/end characters
  for ch1,ch2 in zip(chrs,chrs[1:]):
    ix1,ix2 = stoi[ch1],stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ',num)

# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs.
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)

# %%
# Gradient descent:
for k in range(100):
    # forward pass:
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
    logits = xenc @ W # predict log counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdim=True) # probs of next character

    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())

    # Backward pass:  
    W.grad = None # set grad to zero
    loss.backward()

    # update
    W.data += -50 * W.grad  

# %%
