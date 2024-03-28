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
def team_discretise(state, m, func):
    state_new = np.empty_like(m, dtype=int)
    # Cut in chunks of m and process
    for i in range(len(m)):
        temp = np.array(list(state[m[:i].sum():m[:i+1].sum()]),dtype=int)
        state_new[i] = func(temp)
    # Convert to string
    state_new = state_new.astype(str)
    state_new = ''.join(state_new)
    return state_new

def mean_05(x):
    return (np.mean(x)>=0.5)
#%%
def Fn_team(topo, how='all'):
    # Get the number of nodes and the team size
    n_max = int(topo.split('_')[0].replace('Team',''))
    m = np.array(topo.split('_')[1:], dtype=int)
    if len(m)==1:
        m = np.full(n_max,m)
    elif len(m)!=n_max:
        raise ValueError('File format does not specify team size')
    # Read the output file
    df = pd.read_csv('../Output/'+topo+'_finFlagFreq.csv')
    # Remove non converged states
    df = df[df['flag']==1].reset_index()
    if len(df)==0:
        return
    # Convert states to team wise binary
    df['states'] = df['states'].str.replace("'",'')
    # Group the states into teams and convert to binary
    type_fn = {'all':np.all, 'any':np.any, 'mean':mean_05}
    df['states'] = df['states'].apply(lambda x: team_discretise(x, m, type_fn[how]))
    # Get the digit sum
    df['sum'] = df['states'].apply(lambda x: sum([int(d) for d in x]))
    # groupby the digit sum and sum up Avg0 and sqrt(sum(SD0^2))  
    df = df.groupby(['sum']).agg({'Avg0':'sum','SD0':'sum'}).reset_index()
    # Get sums not present in the data
    n_not = set(range(0,n_max+1)) - set(df['sum'])
    # Add the missing sums as 
    df = df.reindex(range(0,n_max+1), fill_value=0)
    if len(n_not)>0:
        not_idx = df.index[-len(n_not):]
        df.loc[not_idx,'sum'] = list(n_not)
    return df
# %%
def Fn_all(files, how='all', plot = False):
    Fn_all = pd.DataFrame()
    for topo in files:
        print(topo)
        df = Fn_team(topo, how=how)
        if df is None:
            continue
        df['file'] = topo
        df['Teams'] = df['file'].str.extract('(\d+)').astype(int)
        df['Members'] = df['file'].str.split('_',expand=True).iloc[:,1:].astype(int).mean(axis=1)
        if plot:
            plot_Fn(df, topo, suff='_'+how, pfx='Teams/')
        Fn_all = pd.concat([Fn_all,df], axis=0)
    return Fn_all
#%%
# Equal teams
# Discrete team = 1 if all are 1
EqTeam_Fn_all = Fn_all(eq_files, how='all', plot=True)
# Discrete team = 1 if mean of all is >= 0.5
EqTeam_Fn_mean = Fn_all(eq_files, how='mean')
# Discrete team = 1 if any is 1
EqTeam_Fn_any = Fn_all(eq_files, how='any')
# Sanity check
print('All-Mean Equal:', (EqTeam_Fn_all==EqTeam_Fn_mean).any(axis=None))
print('All-Any Equal:', (EqTeam_Fn_all==EqTeam_Fn_any).any(axis=None))
print('Mean-Any Equal:', (EqTeam_Fn_mean==EqTeam_Fn_any).any(axis=None))
EqTeam_Fn_all['Members'] = EqTeam_Fn_all['Members'].astype(int)
# Save the results (only for all)
EqTeam_Fn_all.to_csv('../Analysed_data/Equal_team_Fn_all.csv',index=False)
#%%
# Unequal teams
# Discrete team = 1 if all are 1
UneqTeam_Fn_all = Fn_all(uneq_files, how='all', plot=True)
# Discrete team = 1 if mean of all is >= 0.5
UneqTeam_Fn_mean = Fn_all(uneq_files, how='mean')
# Discrete team = 1 if any is 1
UneqTeam_Fn_any = Fn_all(uneq_files, how='any')
# Sanity check
print('All-Mean Unequal:', (UneqTeam_Fn_all==UneqTeam_Fn_mean).any(axis=None))
print('All-Any Unequal:', (UneqTeam_Fn_all==UneqTeam_Fn_any).any(axis=None))
print('Mean-Any Unequal:', (UneqTeam_Fn_mean==UneqTeam_Fn_any).any(axis=None))
UneqTeam_Fn_all['Members'] = UneqTeam_Fn_all['Members'].astype(int)
# Save the results (only for all)
UneqTeam_Fn_all.to_csv('../Analysed_data/Unequal_team_Fn_all.csv',index=False)
#%%
# Read the results
EqTeam_Fn_all = pd.read_csv('../Analysed_data/Equal_team_Fn_all.csv')
UneqTeam_Fn_all = pd.read_csv('../Analysed_data/Unequal_team_Fn_all.csv')
#%%
# Select only even number of teams
EqAllE = EqTeam_Fn_all[EqTeam_Fn_all['Teams']%2==0]
UnEqAllE = UneqTeam_Fn_all[UneqTeam_Fn_all['Teams']%2==0]
# Select only odd number of teams
EqAllO = EqTeam_Fn_all[EqTeam_Fn_all['Teams']%2==1]
UnEqAllO = UneqTeam_Fn_all[UneqTeam_Fn_all['Teams']%2==1]
#%%
# F1
plot_Fi(EqTeam_Fn_all, 1, suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UneqTeam_Fn_all, 1, suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Fn/2
plot_Fi(EqAllE, 'n_2', suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UnEqAllE, 'n_2', suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Fn/2+1
plot_Fi(EqAllE, 'n_2+1', suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UnEqAllE, 'n_2+1', suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Fn/2-1
plot_Fi(EqAllE, 'n_2-1', suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UnEqAllE, 'n_2-1', suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Fn-1/2
plot_Fi(EqAllO, 'n-1_2', suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UnEqAllO, 'n-1_2', suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Fn+1/2
plot_Fi(EqAllO, 'n+1_2', suff='_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
plot_Fi(UnEqAllO, 'n+1_2', suff='_uneq_all', pfx='Teams/', hue='Members', palette='viridis', x='Teams', legend=False)
# %%
# Plot n wise
for n in UneqTeam_Fn_all['Teams'].unique():
    df = UneqTeam_Fn_all[UneqTeam_Fn_all['Teams']==n]
    plot_Fn(df, f'Team{n}', pfx='Teams/', suff='_uneq_all', hue='Members', palette='viridis', legend=False)
# %%
# Colorbar bcos ....
fig, ax = plt.subplots(figsize=(10, 2))
cmap = plt.cm.viridis
norm = plt.Normalize(EqTeam_Fn_all['Members'].min(), EqTeam_Fn_all['Members'].max())
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
cax=ax, orientation='horizontal')
cb1.set_label('Average Team Size')
plt.tight_layout()
plt.savefig('../figures/Teams/Teams_colorbar.svg')
# %%
