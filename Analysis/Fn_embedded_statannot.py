#%%
import pandas as pd
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import ttest_ind
sns.set_context('poster')
#%%
Fn_all = pd.read_csv('../Analysed_data/embedded_Fn.csv')
#%%
def F_i(Fn_all, i, x):
    if type(i) == int:
        Fi = Fn_all[Fn_all['sum']==i]
        ylab = r'$F('+str(i)+')$'
    elif i == 'n_2':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]/2]
        ylab = r'$F(\frac{n}{2})$'
    elif i == 'n_2+1':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]/2+1]
        ylab = r'$F(\frac{n}{2}+1)$'
    elif i == 'n_2-1':
        Fi = Fn_all[Fn_all['sum'] == Fn_all[x]/2-1]
        ylab = r'$F(\frac{n}{2}-1)$'
    elif i == 'n-1_2':
        Fi = Fn_all[Fn_all['sum'] == (Fn_all[x]-1)/2]
        ylab = r'$F(\frac{n-1}{2})$'
    elif i == 'n+1_2':
        Fi = Fn_all[Fn_all['sum'] == (Fn_all[x]+1)/2]
        ylab = r'$F(\frac{n+1}{2})$' 
    return Fi, ylab
#%%
def plot_Fi_test(Fn_all,i, x='Nodes', y='Avg0', suff='', pfx='', hue=None, test='t-test_ind', **kwargs):
    Fi, ylab = F_i(Fn_all,i,x)
    # Pairs to compare between the hues for a particular x
    pairs = [[(fixed, size1),(fixed, size2)] for size1, size2 in it.combinations(Fi[hue].unique(),2) for fixed in Fi[x].unique()]
    n_hue = len(Fi[hue].unique())
    ax = sns.barplot(x=x,y=y,data=Fi, hue=hue, palette=sns.cubehelix_palette(n_hue),**kwargs)
    ax.set_ylim(0,1.2)
    annot = Annotator(ax,pairs, data=Fi, x=x, y=y, hue=hue)
    annot.configure(test=test, text_format='star', verbose=2, text_offset=0.5, hide_non_significant=True, fontsize=12)
    annot.apply_test()
    annot.annotate()
    ax.get_legend().remove()
    ax.set_ylim(0,1.2)
    plt.ylabel(ylab)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../figures/'+pfx+'F'+str(i)+suff+'.svg')
    plt.clf()
    plt.close()
#%%
Fns = [1,'n_2','n_2+1','n_2-1','n-1_2','n+1_2']
# Plot density wise
for i in Fns:
    for density in Fn_all['Emb_density'].unique():
        Fn = Fn_all[Fn_all['Emb_density']==density]
        pairs = list(it.combinations(Fn['Emb_size'].unique(),2))
        plot_Fi_test(Fn,i, suff=f'_embdens-{density}',hue='Emb_size', pfx='Embedded/', test='t-test_ind')
#%%
# Plot size wise
for i in Fns:
    for size in Fn_all['Emb_size'].unique():
        Fn = Fn_all[Fn_all['Emb_size']==size]
        pairs = list(it.combinations(Fn['Emb_density'].unique(),2))
        plot_Fi_test(Fn,i, suff=f'_embsize-{size}',hue='Emb_density', pfx='Embedded/', test='t-test_ind')
# %%
# Perform ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
# %%
for i in Fns:
    print(f'F{i}')
    Fi, ylab = F_i(Fn_all,i,'Nodes')
    for nodes in Fi['Nodes'].unique():
        Fii = Fi[Fi['Nodes']==nodes]
        model = ols('Avg0 ~ C(Emb_size) + C(Emb_density) + C(Emb_size):C(Emb_density)', data=Fii).fit()
        print(f'Nodes: {nodes}')
        print(sm.stats.anova_lm(model, typ=2))
# %%
