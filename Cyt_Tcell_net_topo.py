#%%
import numpy as np
import pandas as pd
# %%
def ew_gen(df, src_node, wt):
    df = df.copy()
    df.loc[df.iloc[:,0] == src_node, df.columns[2]] *= wt
    return df

def edge_signalling(fname, src_node, cyt_node, ew_max=5, sw_max=5):
    df = pd.read_csv(f'./TOPO/{fname}.topo',sep='\t')
    df[df.columns[2]] = df[df.columns[2]].replace(2,-1).astype(float)
    for ew in range(1,ew_max+1):
        df_i = ew_gen(df, src_node, ew)
        for sw in range(1,sw_max+1):
            df_j = ew_gen(df_i, cyt_node, sw)
            df_j.to_csv(f'./TOPO/{fname}_{src_node}-{ew}_{cyt_node}-{sw}.topo', index=False, sep='\t')
# %%
edge_signalling(fname = 'Cyt_T4_Th1' , src_node = 'TBX21', cyt_node = 'IL12')
# %%
edge_signalling(fname = 'Cyt_T4_Th2', src_node = 'GATA3', cyt_node = 'IL4')
# %%
edge_signalling(fname = 'Cyt_T4_Th17', src_node = 'RORC', cyt_node = 'IL6')
#%%
edge_signalling(fname = 'Cyt_T4_Treg', src_node = 'FOXP3', cyt_node = 'TGFB')
# %%
