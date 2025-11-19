# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from math import ceil

# %% [markdown]
# ## Load Data

# %%
'''Load base vectors.'''
base = []
with open("../tmp/base_sorted.txt", 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        base.append([float(i) for i in line])
base = np.array(base)

'''Load groupings.'''
grouping = []
with open("../tmp/grouping.txt", 'r') as f:
    line = f.readline().strip().split(' ')
    for i in line:
        grouping.append(int(i))

'''Load graph.'''
graph = []
with open("../tmp/graph.txt", 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        graph.append((int(line[0]), int(line[1])))

base, grouping

# %% [markdown]
# ## Plot Graph

# %%
'''Full Figure'''
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

colors = colormaps["nipy_spectral"](np.linspace(0, 1, len(grouping) - 1))
for i in range(len(grouping) - 1):
    l = grouping[i]
    r = grouping[i+1]
    x = base[l:r, 0]
    y = base[l:r, 1]
    z = base[l:r, 2]
    segments = []
    for (j, k) in graph:
        if l <= j < r and l <= k < r:
            segments.append([base[j], base[k]])
    lc = Line3DCollection(segments, colors=colors[i], alpha=0.5)
    ax.add_collection(lc)
    ax.scatter(x, y, z, c=colors[i], alpha=0.8)

fig.show()

input()
