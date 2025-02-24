import numpy as np
from numba import cuda, int32
import math
from itertools import combinations_with_replacement
from tqdm import tqdm
# ---------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------
n = 6 # Number of nodes in Toggle-n network
BATCH_SIZE = 2**20 # Adjust this value based on your GPU memory
# ---------------------------------------------------------------
# CPU helper functions (same as your original code) 
# ---------------------------------------------------------------
def gen_inps(n):
    # Generate all possible inputs (each row is a binary vector of length n)
    inputs = np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)
    inp_sum = np.sum(inputs, axis=1)
    # Choose pairs (i,j) where sum(input[i]) < sum(input[j])
    lower_idx, higher_idx = np.where(inp_sum[:, None] < inp_sum)
    # Keep only those pairs for which every element of inputs[i] <= inputs[j]
    valid = np.all(inputs[lower_idx] <= inputs[higher_idx], axis=1)
    # Return as a 2-row array for convenience: row0 = indices of lower, row1 = indices of higher.
    inp_pair = np.vstack((lower_idx[valid], higher_idx[valid])).astype(np.int32)
    return inputs.astype(np.int32), inp_pair

def generate_rows(n, inputs):
    rows = list(combinations_with_replacement([0,1], n))
    # Find the index of inputs where the row is present
    return np.where(np.all(inputs[:, None] == rows, axis=2))[0].astype(np.int32)

# ---------------------------------------------------------------
# CUDA Kernel: Each thread processes one function index
# ---------------------------------------------------------------
# We assume that the number of inputs is not larger than, say, 64.
# (For n=5, we have 32 inputs.)
@cuda.jit
def check_monotonic_kernel(func_indices, num_inputs, 
                           inp_pair0, inp_pair1, num_pairs,
                           rows, num_rows,count_arr):
    """
    Each thread processes one function index:
      - Converts the function index (an integer) into a bit-vector 'func'
      - Checks if for every (i,j) in inp_pair, we have func[i] <= func[j]
      - If yes, atomically increments a global counter and adds the bits at positions 'rows'
        into the global count_arr.
    """
    pos = cuda.grid(1)
    if pos >= func_indices.shape[0]:
        return

    # Get the function index for this thread.
    func_idx = func_indices[pos]
    
    # Create a local array for the Boolean function values.
    # We assume that num_inputs is small (e.g. 32)
    func = cuda.local.array(64, int32)  # size 64 is a safe upper bound

    # Convert func_idx into its binary representation.
    # We extract bit j by (func_idx >> j) & 1.
    for j in range(num_inputs):
        # Note: using bitwise operations; cast func_idx to unsigned 64-bit.
        func[j] = (func_idx >> j) & 1

    # Check monotonicity using precomputed pairs.
    is_monotonic = True
    for j in range(num_pairs):
        i_lower = inp_pair0[j]
        i_higher = inp_pair1[j]
        if func[i_lower] < func[i_higher]:
            is_monotonic = False
            break

    # If monotonic, update global counter and count_arr.
    if is_monotonic:
        # For each index in rows, atomically add the corresponding bit.
        for j in range(num_rows):
            idx = rows[j]
            cuda.atomic.add(count_arr, j, func[idx])

# ---------------------------------------------------------------
# Host code to launch kernels in batches.
# ---------------------------------------------------------------
if __name__ == '__main__':
    inputs, inp_pair = gen_inps(n-1)
    rows = generate_rows(n-1, inputs)
    
    num_inputs = inputs.shape[0]   # 2^n inputs
    num_pairs = inp_pair.shape[1]    # number of pairs in inp_pair
    num_rows = rows.shape[0]         # for each summing to 0, 1, ..., n
    
    # Transfer constant data to the device.
    d_inp_pair0 = cuda.to_device(inp_pair[0])
    d_inp_pair1 = cuda.to_device(inp_pair[1])
    d_rows = cuda.to_device(rows)
    
    # Allocate device arrays for global counters:
    # 'count_arr' is an array of length num_rows.
    d_count_arr = cuda.to_device(np.zeros(num_rows, dtype=np.int32))
    
    # Number of batches to run to process all functions.
    TOTAL_FUNCS = 2**num_inputs  # = 2^(2^n)
    BATCHES_TO_RUN = math.ceil(TOTAL_FUNCS / BATCH_SIZE)
    
    # Choose CUDA kernel configuration:
    threadsperblock = 256
    blockspergrid = (BATCH_SIZE + (threadsperblock - 1)) // threadsperblock

    # Process functions in batches.
    for batch in tqdm(range(BATCHES_TO_RUN), desc="Processing batches"):
        start = batch * BATCH_SIZE
        end = start + BATCH_SIZE
        if end > TOTAL_FUNCS:
            end = TOTAL_FUNCS
        # Create an array of function indices for this batch.
        func_indices = np.arange(start, end, dtype=np.uint64)
        d_func_indices = cuda.to_device(func_indices)
        
        # Launch the kernel.
        check_monotonic_kernel[blockspergrid, threadsperblock](
            d_func_indices, num_inputs,
            d_inp_pair0, d_inp_pair1, num_pairs,
            d_rows, num_rows,
            d_count_arr
        )
        
        # (Optional) synchronize after each batch.
        cuda.synchronize()

    # Retrieve results from device.
    count_arr = d_count_arr.copy_to_host()

    # Save the results to a file.
    np.savetxt(f"../Output/MBFs_T{n}.txt", count_arr, fmt='%i')
