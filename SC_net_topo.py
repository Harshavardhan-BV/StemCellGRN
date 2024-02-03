#%%
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'./RASSPE/')
import net_gen as ng
#%%
# Toggle n networks
for n in range(2,9):
    ng.toggle_n(n)
    ng.toggle_n(n, Self='SA')
    ng.toggle_n(n, Self='SI')
# %%
# Toggle n teams of m members
for n in range(2,9):
    for m in range(2,11):
        ng.team_n(n,m)
