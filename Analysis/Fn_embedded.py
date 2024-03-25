#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from common_functions import Fn, plot_Fn, plot_Fi
sns.set_context('poster')
#%%
os.makedirs('../figures/Embedded', exist_ok=True)
# list the files
files = glob.glob('../Output/Embedded_T[1-9]_*_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
# %%
def Fn_embedded(topo, n, size):
    # Read the output file
    df = pd.read_csv('../Output/'+topo+'_finFlagFreq.csv')
    # Remove non converged states
    df = df[df['flag']==1].reset_index()
    if len(df)==0:
        return
    # convert states to binary
    df['states'] = df['states'].str.replace("'",'')
    # Get number of nodes 
    n_max = len(df.loc[0,'states'])
    if n_max != n+size:
        print('Value mismatch with', topo)
        return
    # Select only the original nodes: these are at the last n
    df['states'] = df['states'].str[-n:]
    # Get the digit sum
    df['sum'] = df['states'].apply(lambda x: sum([int(d) for d in x]))
    # groupby the digit sum
    df = df.groupby(['sum']).agg({'Avg0':'sum','SD0':'sum','frust0':'mean'}).reset_index()
    # Get sums not present in the data
    n_not = set(range(0,n+1)) - set(df['sum'])
    # Add the missing sums as 
    df = df.reindex(range(0,n+1), fill_value=0)
    if len(n_not)>0:
        df.iloc[-len(n_not):,:].loc[:,'sum'] = list(n_not)
    return df
# %%
All = pd.DataFrame()
for topo in files:
    print(topo)
    # Get the number of nodes and the team size
    n, size, density, i = np.array(topo.replace('T','').split('_')[1:], dtype=int)
    df = Fn_embedded(topo, n, size)
    if df is None:
        continue
    df['file'] = topo
    df['Nodes'] = n
    df['Emb_size'] = size
    df['Emb_density'] = density
    All = pd.concat([All,df], axis=0)
#%%
All['Emb_sizexDensity'] = All['Emb_size'].astype(str) + "_" + All['Emb_density'].astype(str)
#%%
# Sort All by Emb_size and then by Emb_density
All = All.sort_values(by=['Emb_size','Emb_density'])
#%%
All.to_csv('../Analysed_data/embedded_Fn.csv',index=False)
#%%
# Read if already saved
All = pd.read_csv('../Analysed_data/embedded_Fn.csv')
# %%
# Select only even number of nodes
AllE = All[All['Nodes']%2==0]
# Select only odd number of nodes
AllO = All[All['Nodes']%2==1]
#%%
plot_Fi(All,1, suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
plot_Fi(AllE,'n_2', suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
plot_Fi(AllE,'n_2+1', suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
plot_Fi(AllE,'n_2-1', suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
plot_Fi(AllO,'n-1_2', suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
plot_Fi(AllO,'n+1_2', suff='_embedded',hue='Emb_sizexDensity', pfx='Embedded/', legend=False)
#%%
# Plot size wise
for size in All['Emb_size'].unique():
    plot_Fi(All[All['Emb_size']==size],1, suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
    plot_Fi(AllE[AllE['Emb_size']==size],'n_2', suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
    plot_Fi(AllE[AllE['Emb_size']==size],'n_2+1', suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
    plot_Fi(AllE[AllE['Emb_size']==size],'n_2-1', suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
    plot_Fi(AllO[AllO['Emb_size']==size],'n-1_2', suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
    plot_Fi(AllO[AllO['Emb_size']==size],'n+1_2', suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', palette='viridis')
# %%
# Plot density wise
for density in All['Emb_density'].unique():
    plot_Fi(All[All['Emb_density']==density],1, suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
    plot_Fi(AllE[AllE['Emb_density']==density],'n_2', suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
    plot_Fi(AllE[AllE['Emb_density']==density],'n_2+1', suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
    plot_Fi(AllE[AllE['Emb_density']==density],'n_2-1', suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
    plot_Fi(AllO[AllO['Emb_density']==density],'n-1_2', suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
    plot_Fi(AllO[AllO['Emb_density']==density],'n+1_2', suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/')
# %%
# Plot n wise
for n in All['Nodes'].unique():
    df = All[All['Nodes']==n]
    plot_Fn(df, f'T{n}', pfx='Embedded/', suff='_embedded', hue='Emb_sizexDensity',palette=None)
# %%
