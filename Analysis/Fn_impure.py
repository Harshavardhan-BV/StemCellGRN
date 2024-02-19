#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import Fn, plot_Fn, plot_Fi
#%%
os.makedirs('../figures/Impure', exist_ok=True)
#%%
# 5 node impure
# list the files
files = glob.glob('../Output/Impure_5_*_finFlagFreq.csv')
files = [os.path.basename(x).replace('_finFlagFreq.csv','') for x in files]
#%%
All = pd.DataFrame()
for topo in files:
    print(topo)
    df = Fn(topo)
    if df is None:
        continue
    df['file'] = topo
    df['Impure'] = topo.split('_')[2]
    All = pd.concat([All,df], axis=0)
#%%
All['Impure'] = All['Impure'].astype(int)
All.sort_values(by='Impure', inplace=True)
#%%
plot_Fn(All, '5', hue='Impure', pfx='Impure/', suff='_Impure')
# %%
All.to_csv('../Analysed_data/Impure_5.csv',index=False)