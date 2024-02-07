#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import Fn, plot_Fn, plot_Fi
#%%
def Fn_rand(topo):
    randi = glob.glob('../Output/'+topo+'_rand/*/*_finFlagFreq.csv')
    Fn_rand = pd.DataFrame()
    for i in randi:
        i = i.replace('_finFlagFreq.csv','')
        i = i.replace('../Output/','')
        df = Fn(i)
        if df is None:
            continue
        Fn_rand = pd.concat([Fn_rand,df], axis=0)
    Fn_rand['file'] = topo
    Fn_rand['Nodes'] = Fn_rand['file'].str.extract('(\d+)').astype(int)
    Fn_rand['file'].str.replace(r'T[1-9]','',regex=True)
    return Fn_rand
#%%
# list the files 
files = glob.glob('../Output/*_rand')
files = [os.path.basename(x).replace(f'_rand','') for x in files]
#%%
All = pd.DataFrame()
for topo in files:
    print(topo)
    df = Fn_rand(topo)
    if len(df)==0:
        continue
    plot_Fn(df, topo, suff='_rand')
    All = pd.concat([All,df], axis=0)
# %%
plot_Fi(df,1,'_rand')
# %%
plot_Fi(df,'n_2','_rand')
# %%
plot_Fi(df,'n_2+1','_rand')
# %%
plot_Fi(df,'n_2-1','_rand')