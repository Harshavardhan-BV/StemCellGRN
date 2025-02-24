# %%
from itertools import product, combinations_with_replacement
import numpy as np
from multiprocessing import Pool, Value, Array
#%%
# Input 
n = 5  # Number of nodes in Toggle-n network
#%%
def gen_inps(n):
    # Generate all possible inputs
    inputs = np.array(np.meshgrid(*[[0, 1]]*n)).T.reshape(-1, n)
    inp_sum = np.sum(inputs, axis=1)
    # Choose pairs where left < right sum
    lower_sum = np.array(np.where(inp_sum[:, None] < inp_sum))
    # Compute a comparison matrix where each element (i,j) is True if all elements of inputs[i] <= inputs[j]
    comparison = np.all(inputs[lower_sum[0]] <= inputs[lower_sum[1]], axis=1)
    # Select the indices of lower sum pairs where the comparison is True
    inp_pair = lower_sum[:, comparison]
    return inputs, inp_pair

def generate_rows(n):
    rows = list(combinations_with_replacement([0,1], n))
    # Find the index of inputs where the row is present
    return np.where(np.all(inputs[:, None] == rows, axis=2))[0]

def process_function(index, len_inputs):
    # Convert index to binary representation and make it an array
    return np.fromiter(np.binary_repr(index, width=len_inputs), dtype=int)

def is_monotonic(func, inp_pair):
    # Check if func(b1) >= func(b2)
    return (func[inp_pair[0]] >= func[inp_pair[1]]).all()

def add_func_if_monotonic(func_index):
    func = process_function(func_index, len(inputs))
    if is_monotonic(func, inp_pair):
        add = func[rows]
        with counter.get_lock():
            counter.value += 1
        # If monotonic, count 1s and 0s
        with count_arr.get_lock():
            count_arr[:] += add
#%%
inputs, inp_pair = gen_inps(n-1)
#%%
rows = generate_rows(n-1)
all_funcs = range(2**len(inputs))
#%%
counter = Value('i', 0)
count_arr = Array('i', [0]*len(rows))
with Pool(32) as p:
    p.map(add_func_if_monotonic, all_funcs)
print('Total number of MBFs found:',counter.value)
#%%
# Write the results to a file
results = np.array(count_arr[:])
np.savetxt(f"../Output/MBFs_T{n}.txt", results, fmt='%i')
# %%
