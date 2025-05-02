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
Ths = ['Th1','Th2','Th17','Treg']
#%%
def row_to_string(row):
    return f"${''.join(f'{col}^+' if row[col] == 1 else f'{col}^-' for col in row.index)}$"

def plot_f(ss, master_regs, fname):
    ss = ss.copy()
    master_regs = ss.columns.intersection(master_regs)
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
def read_state(fname, master_regs, nodename=None,reindex=False,outpath='../Output'):
    if nodename is None:
        nodename = fname
    nodes = pd.read_csv(f'{outpath}/{nodename}_nodes.txt', header=None)
    ss = pd.read_csv(f'{outpath}/{fname}_finFlagFreq.csv')
    split_ss= ss['states'].str.split('_', expand=True).astype(int)
    split_ss.columns = nodes[0]
    master_regs = [reg for reg in master_regs if reg in split_ss.columns]
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
df = read_state(fname, master_regs)
plot_f(df, master_regs, fname)
# %%
fname = 'TCellDiff'
df = read_state(fname, master_regs)
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
    df = read_state(topo,master_regs, reindex=True)
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
    df = read_state(topo,master_regs, nodename=fname, outpath=f'../Output/{fname}_rand/0.0_1.0')
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
for Ti in range(2,5):
    for j in range(Ti):
        fname = f'Cyt_T{Ti}_{Ths[j]}'
        df = read_state(fname, master_regs)
        plot_f(df, master_regs, fname)
# %%
def edge_signalling(fname, src_node, cyt_node, master_regs, ew_max=10, sw_max=10):
    e_df = []
    for ew in range(1,ew_max+1):
        for sw in range(1,sw_max+1):
            newfname = f'{fname}_{src_node}-{ew}_{cyt_node}-{sw}'
            df = read_state(newfname, master_regs)
            master_regs = [reg for reg in master_regs if reg in df.columns]
            new_df = df[(df[src_node] == 1) & (df[master_regs].drop(columns=[src_node]).sum(axis=1) == 0)]
            e_df.append([ew, sw, new_df['Avg0'].values[0] if not new_df.empty else 0])
    e_df = pd.DataFrame(e_df, columns=['EW','SW','FA1'])
    return e_df

def plot_edge_signalling(df, fname, cmap='viridis'):
    df = df.pivot(index="SW", columns="EW", values="FA1")
    df = df.iloc[::-1]  # Reverse the order of rows to place 1,1 at the bottom
    sns.heatmap(df, cmap=cmap, vmin=0, vmax=1, square=True, annot=df.map(lambda x: '*' if x == 1 else ''), fmt='')
    plt.xlabel('Edge Weight')
    plt.ylabel('Signalling Strength')
    plt.savefig(f'../figures/TCellDiff/F_{fname}_ES.svg')
#%%
df = edge_signalling('Cyt_T4_Th1','TBX21', 'IL12', master_regs)
plot_edge_signalling(df, 'Cyt_T4_Th1', 'Blues')
# %%
df = edge_signalling('Cyt_T4_Th2','GATA3', 'IL4', master_regs)
plot_edge_signalling(df, 'Cyt_T4_Th2', 'Oranges')
# %%
df = edge_signalling('Cyt_T4_Th17','RORC', 'IL6', master_regs)
plot_edge_signalling(df, 'Cyt_T4_Th17', 'Greens')
# %%
df = edge_signalling('Cyt_T4_Treg','FOXP3', 'TGFB', master_regs)
plot_edge_signalling(df,'Cyt_T4_Treg','Reds')
# %%
