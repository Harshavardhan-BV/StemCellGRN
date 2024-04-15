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
# %%
# Toggle n teams of random members with total m x n members
for i in range(5):
    for n in range(2,9):
        for m in range(2,11):
            ng.team_n_rand_split(n,m*n)
# %%
# Embedded toggle n networks
for n in range(2,9):
    for size in range(10,21,5):
        for density in range(2,7,2):
            ng.embedded_toggle(n, size, density, 100)
#%%
# Impure for Toggle 2-5 network
for n in range(2,6):
    m_max = n*(n-1)
    for m in range(m_max+1):
        ng.impure_n(n,m, 100)
# %%
# Impure for Toggle 6 network, 50 each
n = 6
m_max = n*(n-1)
for m in range(m_max+1):
    ng.impure_n(n,m, 50)