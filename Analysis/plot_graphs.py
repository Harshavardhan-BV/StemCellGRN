#%%
import os
import glob
import seaborn as sns
import sys
sys.path.insert(0,'../RASSPE/')
from racipe_analysis import plot_graphTopo, TopoToAdj
sns.set_context('poster')
#%%
os.chdir('../')
#%%
topos = glob.glob('T[1-9]*.topo', root_dir='./TOPO/')
# %%
for topo in topos:
    topo = topo.replace('.topo','')
    print(topo)
    TopoToAdj(topo, plot=True)
    plot_graphTopo(topo)
# %%
