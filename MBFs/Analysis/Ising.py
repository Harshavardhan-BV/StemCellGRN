#%%
import numpy as np
import pandas as pd
import itertools as it
import sys
sys.path.insert(0,'../codes/')
from MBF_Sym_Net import input_pairs, is_monotonic
#%%
n = 2 # Number of nodes in Toggle-n network
#%%
def ST_IO(n):
    # Generate adjacency matrix for a Toggle-n network
    adjMat = -np.ones((n,n), dtype=int) + np.eye(n, dtype=int)
    # Generate all possible initial vectors
    iv = list(it.product([1,-1],repeat=n))
    # Find the input-output pairs for "synchronous update"
    ov = []
    for i in iv:
        sumvec = adjMat@i
        sumvec = np.sign(sumvec)
        # Where sumvec is 0, keep the original value
        # sumvec = np.where(sumvec==0,i,sumvec)
        ov.append(sumvec)
    df = pd.DataFrame(np.append(iv, ov, axis=1), columns=[f'x{i}' for i in range(n)] + [f'y{i}' for i in range(n)])
    return df

def check_function(df):
    # Check if it is a function
    # Groups by input columns and checks if the output column has only one unique value for all inputs
    nuniq = df.groupby(df.columns[:-1].tolist()).nunique()
    return (nuniq == 1).all(axis=1).all()
    
def check_monotone(df):
    # Check if it is monotone
    # If b1 < b2 then f(b1) <= f(b2)
    inputs = df.iloc[:,:-1].values
    output = df.iloc[:,-1].values
    # Limit checking only when b1 < b2
    inp_pair = input_pairs(inputs)
    return True if is_monotonic(output, inp_pair) else False
    
def check_essential(df):
    # Check if it is essential
    # For each input column j, check if the output column depends on input j
    out = df.columns[-1]
    for j in df.columns[:-1]:
        # Groupby not including j
        not_j = df.columns[:-1].difference([j])
        if not_j.empty:
            df_un = df.nunique()
        else: 
            df_un = df.groupby(not_j.tolist()).nunique()
        # If any of the output column changes on change of input j (keeping other inputs fixed), then it is essential
        if not (df_un[out] == df_un[j]).any():
            return False
    return True

def check_embf(df):
    # Check all properties of an Essential Monotone Boolean function with mapping {-1: "Not function", -2: "Not Monotone", -3: "Not Essential", 0: "Essential Monotone Boolean Function"}
    if not check_function(df):
        return -1
    if not check_monotone(df):
        return -2
    if not check_essential(df):
        return -3
    return 0

embf_map = {-1: "Not Function", -2: "Not Monotone", -3: "Not Essential", 0: "Essential Monotone Boolean Function"}
# %%
df = ST_IO(n)
df.to_csv(f'../Output/Ising_T{n}_IO.txt', index=False, sep=' ')
# %%
outs = []
for i in range(n):
    # Select the inputs corresponding to the output y_i
    idx = (df.columns.str.contains('x') & ~df.columns.str.contains(f'x{i}') | df.columns.str.fullmatch(f'y{i}'))
    df_io = df.loc[:, idx]
    df_io = df_io.drop_duplicates()
    # If zeros present in the output consider two cases
    if (df_io.iloc[:,-1] == 0).any():
        for j in [1,-1]:
            df_io1 = df_io.replace(0,j)
            embf = check_embf(df_io1)
            outs.append([f'y{i}', df_io1[f'y{i}'].values, embf])
    else:
        embf = check_embf(df_io)
        outs.append([f'y{i}', df_io[f'y{i}'].values, embf])
# %%
df = pd.DataFrame(outs, columns=['Output_node', 'Output_values', 'EMBF'])
# %%
df['Out_bin'] = df['Output_values'].apply(lambda x: ''.join([str(i).replace('-1', '0') for i in x]))
df['Status'] = df['EMBF'].map(embf_map)
# %%
df.to_csv(f'../Output/Ising_T{n}.csv', index=False)