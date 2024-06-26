#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import Fn, plot_Fn, plot_Fi
sns.set_context('poster')
# %%
def Fn_all(files):
    Fn_all = pd.DataFrame()
    for topo in files:
        print(topo)
        df = Fn(topo)
        if df is None:
            continue
        df['file'] = topo
        df['Nodes'] = df['file'].str.extract('(\d+)').astype(int)
        plot_Fn(df, topo)
        Fn_all = pd.concat([Fn_all,df], axis=0)
    return Fn_all
#%%
All = pd.DataFrame()
#%%
# Without self regulation
# list the files
files = glob.glob('../Output/T[1-9]_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
df = Fn_all(files)
plot_Fi(df,1)
df['Reg'] = 'None'
All = pd.concat([All,df], axis=0)
# %%
# With self activation
files = glob.glob('../Output/T[1-9]SA_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
df = Fn_all(files)
plot_Fi(df,1,suff='_SA')
df['Reg'] = 'SA'
All = pd.concat([All,df], axis=0)
# %%
# With self inhibition
files = glob.glob('../Output/T[1-9]SI_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
df = Fn_all(files)
plot_Fi(df,1,suff='_SI')
df['Reg'] = 'SI'
All = pd.concat([All,df], axis=0)
# %%
All.to_csv('../Analysed_data/nhedron_Fn.csv',index=False)
# %%
plot_Fi(All,1, suff='_hedron',hue='Reg')
#%%
# Select only even number of nodes
AllE = All[All['Nodes']%2==0]
plot_Fi(AllE,'n_2', suff='_hedron',hue='Reg')
plot_Fi(AllE,'n_2+1', suff='_hedron',hue='Reg')
plot_Fi(AllE,'n_2-1', suff='_hedron',hue='Reg')
# %%
# Select only odd number of nodes
AllO = All[All['Nodes']%2==1]
plot_Fi(AllO,'n-1_2', suff='_hedron',hue='Reg')
plot_Fi(AllO,'n+1_2', suff='_hedron',hue='Reg')