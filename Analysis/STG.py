#%%
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../RASSPE/')
from racipe_analysis import TopoToAdj
os.chdir('../')
plt.rcParams['svg.hashsalt'] = ''
sns.set_context('poster')
#%%
topo = 'T3SI'
ew = True
#%%
def STG(adjMat):
    n = len(adjMat)
    # Generate all possible combinations of initial vectors
    iv = list(it.product([1,-1],repeat=n))
    # Create a matrix with iv as start and end
    states = pd.DataFrame(np.zeros((len(iv),len(iv))),index=iv, columns=iv)
    for i in iv:
        sumvec = adjMat@i
        sumvec = np.sign(sumvec)
        # Where sumvec is 0, keep the original value
        sumvec = np.where(sumvec==0,i,sumvec)
        for j, k in enumerate(sumvec):
            i_new = np.array(i)
            i_new[j] = k
            i_new = tuple(i_new.tolist())
            states.loc[[i],[i_new]] += 1
    states = states.div(states.sum(axis=1),axis=1)
    # Convert to string format and replace -1 with 0
    states.index = states.index.map(lambda x: "".join(map(str, x)).replace('-1','0') )
    states.columns = states.columns.map(lambda x: "".join(map(str, x)).replace('-1','0') )
    return states

def STG_topo(topo, ew=False):
    adjMat = TopoToAdj(topo, plot=False)
    if ew:
        for i in range(100):
            adjMat1 = np.random.random(size=adjMat.shape) * adjMat
            states = STG(adjMat1)
            if any(states.eq(1).any()):
                adjMat = adjMat1
                break
    else:
        states = STG(adjMat)
    return states, adjMat

def plot_STG(states, topo, suff=False, seed=0):
    STGraph = nx.from_pandas_adjacency(states, create_using=nx.DiGraph)
    # Rename the nodes and remove the '(' and ')' characters
    STGraph = nx.relabel_nodes(STGraph, lambda x: str(x).replace('(','').replace(')',''))
    plt.figure(figsize=(5,5))
    pos = nx.spring_layout(STGraph, seed=seed)
    edges = STGraph.edges(data=True)
    edge_colors = [d['weight'] for (u, v, d) in edges]
    nx.draw_networkx_nodes(STGraph, pos, node_size=700, node_color='skyblue', margins=0.1)
    nx.draw_networkx_labels(STGraph, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(STGraph, pos, edge_color=edge_colors, edge_cmap=plt.cm.Greys, edge_vmin=0, edge_vmax=1, node_size=1000, arrowsize=10, connectionstyle='arc3,rad=0.2')
    suff = '-ew' if suff else ''
    plt.title(f'{topo}{suff}', fontsize=15)
    plt.savefig(f'./figures/STG-{topo}{suff}.svg')

def plot_adj(adjMat, topo):
    sns.heatmap(adjMat, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidth=2)
    plt.savefig(f'./figures/AdjMat_{topo}-ew.svg')
    plt.clf()
    plt.close()
# %%
states, adjMat = STG_topo(topo, ew)
# %%
plot_STG(states, topo, suff=ew, seed=20)
# %%
if ew:
    plot_adj(adjMat, topo)
# %%
