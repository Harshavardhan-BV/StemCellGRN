#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import Fn, plot_Fn, plot_Fi
sns.set_context('poster')
#%%
os.makedirs('../figures/Impure', exist_ok=True)
#%%
# impure upto 5 nodes
# list the files
files = glob.glob('../Output/Impure_*_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
#%%
All = pd.DataFrame()
for topo in files:
    print(topo)
    df = Fn(topo)
    if df is None:
        continue
    df['file'] = topo
    df['Nodes'] = topo.split('_')[1]
    df['Impure'] = topo.split('_')[2]
    All = pd.concat([All,df], axis=0)
#%%
All['Nodes'] = All['Nodes'].astype(int)
All['Impure'] = All['Impure'].astype(int)
All.sort_values(by=['Nodes','Impure'], inplace=True)
# %%
All.to_csv('../Analysed_data/Impure.csv',index=False)
#%%
All = pd.read_csv('../Analysed_data/Impure.csv')
#%%
for i in All.Nodes.unique():
    plot_Fn(All[All['Nodes']==i], f'{i}-node', hue='Impure', pfx='Impure/', suff='_Impure', legend=False, palette='rocket')
# %%
def bar_impure(All,n):
    sns.lineplot(x='Impure',y='Avg0',data=All, palette='rocket', hue='sum', legend=False, err_style='bars', err_kws={'capsize':5}, dashes=True)
    plt.title(f'{n}-node')
    plt.xlabel('Impurity')
    plt.ylabel('Frequency')
    plt.ylim(0,1.1)
    # plt.xticks(rotation=)
    plt.savefig(f'../figures/Impure/Impure_{n}.svg')
    plt.clf()
    plt.close()
# %%
for i in All.Nodes.unique():
    bar_impure(All[All['Nodes']==i],i)
# %%
