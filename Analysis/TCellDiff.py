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
def read_state(fname, nodename=None,reindex=False,outpath='../Output'):
    if nodename is None:
        nodename = fname
    nodes = pd.read_csv(f'{outpath}/{nodename}_nodes.txt', header=None)
    ss = pd.read_csv(f'{outpath}/{fname}_finFlagFreq.csv')
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
# %%
files = glob.glob('Embedded_TCellDiff_*_finFlagFreq.csv',root_dir='../Output')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
# %%
All = pd.DataFrame()
for topo in files:
    print(topo)
    # Get the number of nodes and the team size
    size, density, i = np.array(topo.split('_')[2:], dtype=int)
    df = read_state(topo, reindex=True)
    if df is None:
        continue
    df['file'] = topo
    df['Emb_size'] = size
    df['Emb_density'] = density
    All = pd.concat([All,df], axis=0)
#%%
All.to_csv('../Analysed_data/Embedded_TCellDiff.csv',index=False)
# %%
plot_f(All, master_regs,f'Embedded_{fname}')
#%%
for Emb_size in All['Emb_size'].unique():
    for Emb_dens in All['Emb_density'].unique():
        All_i = All[(All['Emb_density']==Emb_dens) & (All['Emb_size']==Emb_size)]
        plot_f(All_i,master_regs,f'Embedded_{fname}_embsize-{Emb_size}_embdens-{Emb_dens}')
# %%
files = glob.glob(f'*_finFlagFreq.csv',root_dir=f'../Output/{fname}_rand/0.0_1.0')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
# %%
All = pd.DataFrame()
for topo in files:
    print(topo)
    df = read_state(topo, nodename=fname, outpath=f'../Output/{fname}_rand/0.0_1.0')
    if df is None:
        continue
    df['file'] = topo
    All = pd.concat([All,df], axis=0)
# %%
All.to_csv('../Analysed_data/TCellDiff_rand.csv',index=False)
#%%
All = All[All[master_regs].apply(tuple, axis=1).isin(All.groupby(master_regs).any()['Avg0'].index)]
# %%
plot_f(All, master_regs,f'{fname}_rand')
# %%
