# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from math import ceil

# %% [markdown]
# ## Load Data

# %%
'''Load base vectors.'''
base = []
with open("../base_sorted.txt", 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        base.append([float(i) for i in line])
base = np.array(base)

'''Load split indicies.'''
split_indicies = [0]
with open("../split_indicies.txt", 'r') as f:
    line = f.readline().strip().split(' ')
    for i in line:
        split_indicies.append(int(i))
split_indicies.append(len(base))

base, split_indicies

# %% [markdown]
# ## Plot Graph

# %%
'''Full Figure'''
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

colors = colormaps["nipy_spectral"](np.linspace(0, 1, len(split_indicies) - 1))
for i in range(len(split_indicies) - 1):
    l = split_indicies[i]
    r = split_indicies[i+1]
    x = base[l:r, 0]
    y = base[l:r, 1]
    z = base[l:r, 2]
    ax.scatter(x, y, z, c=colors[i], alpha=0.8)

fig.show()

input()

