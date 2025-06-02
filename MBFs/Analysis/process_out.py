#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.hashsalt'] = ''
sns.set_context('poster')
#%%
lis = []
for i in range(2, 7):
    df = pd.read_csv(f"../Output/MBFs_T{i}.txt", header=None)
    df.columns = ['v']
    for j in range(i+1):
        if j==0:
            log_lhs = 0
        else:
            log_lhs = j * np.log(df.loc[j-1, 'v'])
        if j==i:
            log_rhs = 0
        else:
            log_rhs = (i-j) * np.log(df.loc[(i-j)-1, 'v'])
        nij = np.exp(log_lhs + log_rhs)
        lis.append([i, j, nij])
# %%
df = pd.DataFrame(lis, columns=['n', 'k', 'phi'])
df.to_csv("../Analysed_data/MBFs.csv", index=False)
# %%
p = sns.barplot(x='n', y='phi', data=df, hue='k', palette='rocket')
plt.yscale('log')
plt.ylabel(r'$\phi^n_k$')
plt.savefig(f"../figures/MBFs.svg", bbox_inches='tight')
# %%
for i in df['i'].unique():
    df1 = df[df['i']==i]
    plt.figure()
    sns.barplot(x='j', y='nij', data=df1, palette='rocket', hue='j', legend=False)
    plt.savefig(f"../figures/MBFs_T{i}.svg")
# %%
