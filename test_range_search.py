#!/usr/bin/env python3
import subprocess
import numpy as np
from scipy.io import savemat, loadmat
import os

# Prepend the MATLAB directory to the PATH
os.environ["PATH"] = r"D:\\Matlab2024a\\bin\\win64;" + os.environ["PATH"]

# Parameters
N = 200     # Number of dataset binary codes
NQ = 100     # Number of query points
B = 64       # Number of bits per code (B must be a multiple of 8)
K = 5        # Number of nearest neighbors
m = 8        # Number of hash tables for MIH
range_threshold = 5  # Fixed range threshold for range search

# Generate random binary dataset and queries
dataset = np.random.randint(0, 2, (N, B), dtype=np.uint8)
queries = np.random.randint(0, 2, (NQ, B), dtype=np.uint8)

# Pack bits (store in bytes to match MIH's expected format)
# Note: For MATLAB input, we need to save the matrix so that each column represents a code.
dataset_packed = np.packbits(dataset, axis=1)   # shape: (N, B/8)
queries_packed = np.packbits(queries, axis=1)       # shape: (NQ, B/8)
print("Dataset shape:", dataset_packed.shape)

# Save in MATLAB .mat format.
# Transpose the packed arrays so that they become (B/8, N) and (B/8, NQ) respectively,
# matching the C++ code expectations in mih_interface.cpp.
dataset_file = "dataset_test.mat"
mih_knn_output_file = "mih_knn_results.mat"
# mih_range_output_file = "mih_range_results.mat"  # For range search if needed

savemat(dataset_file, {"B": dataset_packed.T, "Q": queries_packed.T})

# Define full paths for the MIH executable and dataset file.
exe_path = os.path.join(os.getcwd(), "build", "mih.exe")
dataset_path = os.path.join(os.getcwd(), dataset_file)

# Construct the MIH command for k‑NN search (without -r)
mih_knn_command = [
    exe_path,
    dataset_path,
    mih_knn_output_file,
    # "-B", str(B),
    "-m", str(m),
    "-Q", str(NQ),
    "-K", str(K)
]

# Remove previous output file if it exists
if os.path.exists(mih_knn_output_file):
    os.remove(mih_knn_output_file)
    
print("Running k‑NN command:", " ".join(mih_knn_command))
exit_code_knn = subprocess.run(mih_knn_command).returncode
print("k‑NN Exit Code:", exit_code_knn)

def load_mih_results(output_file):
    """
    Load MIH results from the MATLAB .mat output file.
    The file is expected to contain a variable 'ret' which is a structure array.
    The element corresponding to the current value of K (i.e. at index K-1) is used.
    
    From this structure element, the 'res' field (a numeric matrix of size (K, NQ))
    and the 'nres' field (range search results) are extracted.
    The 'res' entries are assumed to be 1‑based indices (as produced by the C++ code)
    and are converted to 0‑based indices.
    """
    mat_data = loadmat(output_file, squeeze_me=True, struct_as_record=False)
    if "ret" not in mat_data:
        raise KeyError("Variable 'ret' not found in the .mat file.")
    
    ret = mat_data["ret"]
    # Extract the structure corresponding to the current K (assuming ret is a 1D array)
    element = ret[K-1]
    
    # Check for expected fields 'res' and 'nres'
    if not hasattr(element, 'res') or not hasattr(element, 'nres'):
        raise KeyError("Expected fields 'res' and 'nres' not found in ret struct.")
    
    # 'res' is a (K x NQ) matrix where each column represents the k‑NN results for a query.
    knn_results = element.res
    knn_results_list = []
    # Convert each column into a set of 0‑based indices.
    for col in range(knn_results.shape[1]):
        indices = knn_results[:, col].astype(np.int64) - 1  # adjust from 1‑based to 0‑based
        knn_results_list.append(set(indices))
    
    # Process range search results; if nres is 2D, take the first column.
    nres_data = element.nres
    if np.ndim(nres_data) == 2:
        range_results = nres_data[:, 0].tolist()
    else:
        range_results = nres_data.tolist()
    return knn_results_list, range_results

# Load MIH results for k‑NN.
# (The range search part is omitted or can be enabled similarly.)
mih_knn, nres_knn = load_mih_results(mih_knn_output_file)

# Compute ground‑truth k‑NN using exact bit comparisons (Hamming distance)
def compute_knn(dataset, queries, K):
    knn_results = []
    for q in dataset:  # Use each code in the original un-packed dataset
        dists = np.array([np.sum(q != d) for d in dataset])
        knn_results.append(set(np.argsort(dists)[:K]))
    return knn_results

gt_knn = compute_knn(dataset, queries, K)

# Compare k‑NN results as sets
knn_match = all(mih_knn[i] == gt_knn[i] for i in range(NQ))
print("k‑NN results match:", knn_match)

if not knn_match:
    print("Differences in k‑NN results!")
    print("MIH k‑NN results (sorted):", [sorted(list(s)) for s in mih_knn[:5]])
    print("Ground truth k‑NN (sorted):", [sorted(list(s)) for s in gt_knn[:5]])
