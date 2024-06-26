import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.hashsalt'] = ''

def Fn(topo):
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
    # Get the digit sum
    df['sum'] = df['states'].apply(lambda x: sum([int(d) for d in x]))
    # groupby the digit sum
    df = df.groupby(['sum']).agg({'Avg0':'sum','SD0':'sum','frust0':'mean'}).reset_index()
    # Get sums not present in the data
    n_not = set(range(0,n_max+1)) - set(df['sum'])
    # Add the missing sums as 
    df = df.reindex(range(0,n_max+1), fill_value=0)
    if len(n_not)>0:
        not_idx = df.index[-len(n_not):]
        df.loc[not_idx,'sum'] = list(n_not)
    return df

def plot_Fn(Fn, topo, suff='', pfx='', hue='sum', palette='rocket',legend=False,**kwargs):
    # Plot the results
    n_max = Fn['sum'].max()
    sns.barplot(x='sum',y='Avg0',data=Fn, order=range(0,n_max+1), hue=hue, palette=palette, legend=legend, **kwargs)
    plt.title(topo)
    plt.xlabel(r'$n_{high}$')
    plt.ylabel('Frequency')
    plt.savefig('../figures/'+pfx+'Fn_'+topo+suff+'.svg')
    plt.clf()
    plt.close()

def plot_Fi(Fn_all,i, x='Nodes', y='Avg0', suff='', pfx='', hue=None, **kwargs):
    if type(i) == int:
        Fi = Fn_all[Fn_all['sum']==i]
        ylab = r'$F('+str(i)+')$'
    elif i == 'n_2':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]//2]
        ylab = r'$F(\frac{n}{2})$'
    elif i == 'n_2+1':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]//2+1]
        ylab = r'$F(\frac{n}{2}+1)$'
    elif i == 'n_2-1':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]//2-1]
        ylab = r'$F(\frac{n}{2}-1)$'
    elif i == 'n-1_2':
        Fi = Fn_all[Fn_all['sum'] == (Fn_all[x]-1)//2]
        ylab = r'$F(\frac{n-1}{2})$'
    elif i == 'n+1_2':
        Fi = Fn_all[Fn_all['sum'] == (Fn_all[x]+1)//2]
        ylab = r'$F(\frac{n+1}{2})$'
    sns.barplot(x=x,y=y,data=Fi, hue=hue, **kwargs)
    plt.ylim(0,1.1)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig('../figures/'+pfx+'F'+str(i)+suff+'.svg')
    plt.clf()
    plt.close()