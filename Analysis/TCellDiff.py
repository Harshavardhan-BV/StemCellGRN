#%%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
plt.rcParams['svg.hashsalt'] = ''
sns.set_context('poster', font_scale=0.6)
#%%
master_regs = ['TBX21','GATA3','RORC','FOXP3','BCL6']
#%%
def row_to_string(row):
    return f"${''.join(f'{col}^+' if row[col] == 1 else f'{col}^-' for col in row.index)}$"

def plot_f(ss, master_regs, fname):
    ss = ss.copy()
    ss['n_high'] = ss[master_regs].sum(axis=1)
    ss['Decimal'] = ss[master_regs].apply(lambda row: int(''.join(map(str, row)), 2), axis=1)
    ss = ss.sort_values(['n_high', 'Decimal'], ascending=[True, False])
    ss['State'] = ss[master_regs].apply(row_to_string, axis=1)
    plt.figure(figsize=(6.4, 1.6+0.4*len(ss['State'].unique())))  # Increased figure size
    sns.barplot(ss, x='Avg0', y='State', hue='n_high', palette=sns.color_palette('rocket', n_colors=len(master_regs)+1), hue_order=range(len(master_regs)+1), legend=False)
    plt.tight_layout()
    plt.xlabel('Frequency')
    plt.savefig(f'../figures/TCellDiff/F_{fname}.svg')
#%%
def read_state(fname, reindex=False):
    nodes = pd.read_csv(f'../Output/{fname}_nodes.txt', header=None)
    ss = pd.read_csv(f'../Output/{fname}_finFlagFreq.csv')
    split_ss= ss['states'].str.split('_', expand=True).astype(int)
    split_ss.columns = nodes[0]
    split_ss = split_ss[master_regs]
    ss = pd.concat((split_ss, ss),axis=1)
    # Remove non converged states
    ss = ss[ss['flag']==1].reset_index()
    if len(ss)==0:
        return
    if reindex:
        ss = ss.set_index(master_regs)
        ss = ss.groupby(level=master_regs).mean(numeric_only=True)
        ss = ss.reindex(index=pd.MultiIndex.from_tuples(it.product([1, 0], repeat=len(master_regs)), names=ss.index.names), fill_value=0)
        ss = ss.reset_index()
    return ss
# %%
fname = 'TCellDiff_SA'
df = read_state(fname)
plot_f(df, master_regs, fname)
# %%
fname = 'TCellDiff'
df = read_state(fname)
plot_f(df, master_regs, fname)
#%%
