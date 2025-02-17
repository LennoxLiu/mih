#!/usr/bin/env python3
import subprocess 
import numpy as np
import h5py
import os

# Parameters
N = 10000      # Number of dataset binary codes
NQ = 100      # Number of query points
B = 128       # Number of bits per code
K = 6         # Number of nearest neighbors
m = 8         # Number of hash tables for MIH
range_threshold = 64  # Fixed range threshold for range search

# Generate random binary dataset and queries
dataset = np.random.randint(0, 2, (N, B), dtype=np.uint8)
queries = np.random.randint(0, 2, (NQ, B), dtype=np.uint8)

# Pack bits (store in bytes to match MIH's expected format)
dataset_packed = np.packbits(dataset, axis=1)
queries_packed = np.packbits(queries, axis=1)

# Save in HDF5 format
dataset_file = "dataset_test.h5"
mih_knn_output_file = "mih_knn_results.h5"
mih_range_output_file = "mih_range_results.h5"

with h5py.File(dataset_file, "w") as f:
    f.create_dataset("B", data=dataset_packed)
    f.create_dataset("Q", data=queries_packed)

# Define full paths for the MIH executable and dataset file.
exe_path = os.path.join(os.getcwd(), "build", "mih")
dataset_path = os.path.join(os.getcwd(), dataset_file)

# Construct the MIH command for k‑NN search (without -r)
mih_knn_command = [
    exe_path,
    dataset_path,
    mih_knn_output_file,
    "-N", str(N),
    "-B", str(B),
    "-m", str(m),
    "-Q", str(NQ),
    "-K", str(K)
]

# Construct the MIH command for range search (with -r)
mih_range_command = [
    exe_path,
    dataset_path,
    mih_range_output_file,
    "-N", str(N),
    "-B", str(B),
    "-m", str(m),
    "-Q", str(NQ),
    "-K", str(K),
    "-r", str(range_threshold)
]

print("Running k‑NN command:", " ".join(mih_knn_command))
exit_code_knn = subprocess.run(mih_knn_command).returncode
print("k‑NN Exit Code:", exit_code_knn)

print("Running range search command:", " ".join(mih_range_command))
exit_code_range = subprocess.run(mih_range_command).returncode
print("Range search Exit Code:", exit_code_range)

def load_mih_results(output_file):
    """
    Load MIH results from the output HDF5 file.
    This function expects two datasets under the 'refs' group:
      - '<prefix>.res' : k‑NN results (each row is a list of dataset indices, 1‑based).
      - '<prefix>.nres': range search results.
    
    For range search results, if nres has two dimensions we take the first column as the count.
    """
    with h5py.File(output_file, "r") as f:
        if "refs" not in f:
            raise KeyError("No 'refs' key found in HDF5 file.")
        
        # Identify MIH result keys (e.g. "mih0.res", "mih0.nres", etc.)
        mih_keys = [key for key in f["refs"].keys() 
                    if key.startswith("mih") and key[3:].split('.')[0].isdigit()]
        if not mih_keys:
            raise KeyError("No valid MIH result keys found in HDF5 file.")
        
        mih_index = max(int(k[3:].split('.')[0]) for k in mih_keys)
        mih_prefix = f"mih{mih_index}"
        res_key = f"{mih_prefix}.res"
        nres_key = f"{mih_prefix}.nres"
        
        if res_key not in f["refs"] or nres_key not in f["refs"]:
            raise KeyError(f"Expected datasets not found: {res_key} or {nres_key}")
        
        # Load k‑NN results (adjust indices from 1‑based to 0‑based)
        knn_results = [set(np.asarray(row, dtype=np.int64) - 1) 
                       for row in f["refs"][res_key][()]]
        
        # Load range search results
        nres_data = f["refs"][nres_key][()]
        if len(nres_data.shape) == 2:
            # Assume the first column is the range count.
            range_results = nres_data[:, 0].tolist()
        else:
            range_results = nres_data.tolist()
    return knn_results, range_results

# Load MIH results for k‑NN and range search.
mih_knn, nres_knn = load_mih_results(mih_knn_output_file)
mih_range_knn, range_counts = load_mih_results(mih_range_output_file)

# Compute ground‑truth k‑NN using exact bit comparisons (Hamming distance)
def compute_knn(dataset, queries, K):
    knn_results = []
    for q in queries:
        dists = np.array([np.sum(q != d) for d in dataset])
        knn_results.append(set(np.argsort(dists)[:K]))
    return knn_results

gt_knn = compute_knn(dataset, queries, K)

# Compute ground‑truth range counts using exact bit comparisons
def compute_range_counts(dataset, queries, R):
    counts = []
    for q in queries:
        dists = np.array([np.sum(q != d) for d in dataset])
        counts.append(int(np.sum(dists < R)))
    return counts

gt_range = compute_range_counts(dataset, queries, range_threshold)

# Compare k‑NN results as sets
knn_match = all(mih_knn[i] == gt_knn[i] for i in range(NQ))
# Compare range counts as scalars
range_match = all(range_counts[i] == gt_range[i] for i in range(NQ))

print("k‑NN results match:", knn_match)
print("Range search counts match:", range_match)

if not knn_match:
    print("Differences in k‑NN results!")
    print("MIH k‑NN results (sorted):", [sorted(list(s)) for s in mih_knn[:5]])
    print("Ground truth k‑NN (sorted):", [sorted(list(s)) for s in gt_knn[:5]])
if not range_match:
    print("Differences in range search counts!")
    print("MIH range counts:", range_counts[:5])
    print("Ground truth range counts:", gt_range[:5])
