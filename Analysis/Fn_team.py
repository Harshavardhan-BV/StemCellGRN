#%%
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import Fn, plot_Fn, plot_Fi
#%%
os.makedirs('../figures/Teams', exist_ok=True)

# list the files
files = glob.glob('../Output/Team[1-9]_*_finFlagFreq.csv')
eq_files = []
uneq_files = []
for file in files:
    file = os.path.basename(file).replace('_finFlagFreq.csv','')
    if file.count('_')>1:
        # Unequal teams
        uneq_files.append(file)
    else:
        # Equal teams
        eq_files.append(file)
#%%
def team_discretise_all(state, m):
    # Split the string into m each with numpy
    state = np.array(list(state),dtype=int).reshape(-1,m)
    # Check if all are 1
    state = np.all(state,axis=1).astype(int)
    # Convert to string
    state = state.astype(str)
    state = ''.join(state)
    return state
def team_discretise_mean(state, m, threshold=0.5):
    # Split the string into m each with numpy
    state = np.array(list(state),dtype=int).reshape(-1,m)
    # Check if mean is >= threshold
    state = np.mean(state,axis=1)
    state = (state>=threshold).astype(int)
     # Convert to string
    state = state.astype(str)
    state = ''.join(state)
    return state
def team_discretise_any(state, m):
    # Split the string into m each with numpy
    state = np.array(list(state),dtype=int).reshape(-1,m)
    # Check if any is 1
    state = np.any(state,axis=1).astype(int)
    # Convert to string
    state = state.astype(str)
    state = ''.join(state)
    return state
#%%
def Fn_team(topo, type='all'):
    # Get the number of nodes and the team size
    n_max = int(topo.split('_')[0].replace('Team',''))
    m = int(topo.split('_')[1])
    # Read the output file
    df = pd.read_csv('../Output/'+topo+'_finFlagFreq.csv')
    # Remove non converged states
    df = df[df['flag']==1]
    if len(df)==0:
        return
    # Convert states to team wise binary
    df['states'] = df['states'].str.replace("'",'')
    # Group the states into teams and convert to binary
    if type=='all':
        df['states'] = df['states'].apply(lambda x: team_discretise_all(x,m))
    elif type=='mean':
        df['states'] = df['states'].apply(lambda x: team_discretise_mean(x,m))
    elif type=='any':
        df['states'] = df['states'].apply(lambda x: team_discretise_any(x,m))
    else:
        raise ValueError('type should be all or mean')
    # Get the digit sum
    df['sum'] = df['states'].apply(lambda x: sum([int(d) for d in x]))
    # groupby the digit sum and sum up Avg0 and sqrt(sum(SD0^2))  
    df = df.groupby(['sum']).agg({'Avg0':'sum','SD0':'sum'}).reset_index()
    # Get sums not present in the data
    n_not = set(range(0,n_max+1)) - set(df['sum'])
    # Add the missing sums as 
    df = df.reindex(range(0,n_max+1), fill_value=0)
    if len(n_not)>0:
        df.iloc[-len(n_not):,:].loc[:,'sum'] = list(n_not)
    return df
# %%
# Equal teams
# Discrete team = 1 if all are 1
Fn_all = pd.DataFrame()
for topo in eq_files:
    print(topo)
    df = Fn_team(topo, type='all')
    if df is None:
        continue
    df['file'] = topo
    df['Nodes'] = df['file'].str.extract('(\d+)').astype(int)
    df['Members'] = df['file'].str.extract('_(\d+)').astype(int)
    plot_Fn(df, topo, suff='_all', pfx='Teams/')
    Fn_all = pd.concat([Fn_all,df], axis=0)
# %%
plot_Fi(Fn_all,1, suff='_team_all',hue='Members', pfx='Teams/')
plot_Fi(Fn_all,'n_2', suff='_team_all',hue='Members', pfx='Teams/')
plot_Fi(Fn_all,'n_2+1', suff='_team_all',hue='Members', pfx='Teams/')
plot_Fi(Fn_all,'n_2-1', suff='_team_all',hue='Members', pfx='Teams/')
#%%
Fn_all.to_csv('../Analysed_data/team_Fn_all.csv',index=False)
# %%
# Discrete team = 1 if mean of all is >= 0.5
Fn_mean = pd.DataFrame()
for topo in eq_files:
    print(topo)
    df = Fn_team(topo, type='mean')
    if df is None:
        continue
    df['file'] = topo
    df['Nodes'] = df['file'].str.extract('(\d+)').astype(int)
    df['Members'] = df['file'].str.extract('_(\d+)').astype(int)
    plot_Fn(df, topo, suff='_mean', pfx='Teams/')
    Fn_mean = pd.concat([Fn_mean,df], axis=0)
# %%
plot_Fi(Fn_mean,1, suff='_team_mean',hue='Members', pfx='Teams/')
plot_Fi(Fn_mean,'n_2', suff='_team_mean',hue='Members', pfx='Teams/')
plot_Fi(Fn_mean,'n_2+1', suff='_team_mean',hue='Members', pfx='Teams/')
plot_Fi(Fn_mean,'n_2-1', suff='_team_mean',hue='Members', pfx='Teams/')
#%%
Fn_mean.to_csv('../Analysed_data/team_Fn_mean.csv',index=False)
# %%
# Discrete team = 1 if any is 1
Fn_any = pd.DataFrame()
for topo in eq_files:
    print(topo)
    df = Fn_team(topo, type='any')
    if df is None:
        continue
    df['file'] = topo
    df['Nodes'] = df['file'].str.extract('(\d+)').astype(int)
    df['Members'] = df['file'].str.extract('_(\d+)').astype(int)
    plot_Fn(df, topo, suff='_any', pfx='Teams/')
    Fn_any = pd.concat([Fn_any,df], axis=0)
# %%
plot_Fi(Fn_any,1, suff='_team_any',hue='Members', pfx='Teams/')
plot_Fi(Fn_any,'n_2', suff='_team_any',hue='Members', pfx='Teams/')
plot_Fi(Fn_any,'n_2+1', suff='_team_any',hue='Members', pfx='Teams/')
plot_Fi(Fn_any,'n_2-1', suff='_team_any',hue='Members', pfx='Teams/')
#%%
Fn_any.to_csv('../Analysed_data/team_Fn_any.csv',index=False)