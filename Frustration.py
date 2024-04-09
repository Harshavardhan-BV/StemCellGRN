#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
#%%
def frustration(n, Self=0):
    # Generate the adjacency matrix
    Adj = np.full((n,n),-1)
    np.fill_diagonal(Adj,Self)
    frust=[]
    for Fn in range(0,n+1):
        # Start with all low
        State = np.full(n,-1)
        # Symmetric networks so setting the first Fn to 1
        State[0:Fn] = 1
        # Find the frustrated edges
        product_matrix = Adj * np.outer(State, State)
        # Find the indices where the product is -1
        frusti = np.argwhere(product_matrix == -1)
        edgi = np.argwhere(Adj != 0)
        # Find the frustration
        frust_n = len(frusti)/len(edgi)
        frust.append([n,Self,Fn,frust_n])
    frust = pd.DataFrame(frust, columns=['n','Self','Fn','Frustration'])
    return frust
#%%
def barplot_frust(frust, n, Self):
    sns.barplot(data=frust[(frust['n']==n) & (frust['Self']==Self)], x='Fn', y='Frustration', palette='inferno')
    plt.ylim(0,1)
    Self = ['SI','','SA'][Self+1]
    # plt.axhline(0.5, color='black', linestyle='--')
    plt.title(f'T{n}{Self}')
    plt.savefig(f'./figures/Frustration/Frustration_T{n}{Self}.svg')
    plt.clf()
    plt.close()
#%%
frust = pd.DataFrame()
for n in range(2,9):
    for Self in [0,1,-1]:
        df = frustration(n, Self)
        frust = pd.concat([frust,df], axis=0)
#%%
frust.to_csv('./Analysed_data/Frustration.csv',index=False)
# %%
os.makedirs('./figures/Frustration', exist_ok=True)
for n in range(2,9):
    for Self in [0,1,-1]:
        barplot_frust(frust, n, Self)
# %%
