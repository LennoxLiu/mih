import subprocess 
import numpy as np
import h5py
import os
from scipy.spatial.distance import hamming

# Parameters
N = 1000      # Number of dataset binary codes
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
mih_output_file = "mih_results.h5"

with h5py.File(dataset_file, "w") as f:
    f.create_dataset("B", data=dataset_packed)
    f.create_dataset("Q", data=queries_packed)

# Define full paths
exe_path = os.path.join(os.getcwd(), "build", "mih")
dataset_path = os.path.join(os.getcwd(), dataset_file)
output_path = os.path.join(os.getcwd(), mih_output_file)

# Construct command as a list (now passing the range threshold after "-r")
mih_command = [
    exe_path,
    dataset_path,
    output_path,
    "-N", str(N),
    "-B", str(B),
    "-m", str(m),
    "-Q", str(NQ),
    "-K", str(K),
    "-r", str(range_threshold)
]

# Print for debugging
print("Running command:", " ".join(mih_command))

# Run the command with subprocess
exit_code = subprocess.run(mih_command).returncode

print("Exit Code:", exit_code)
mih_success = exit_code == 0


def load_mih_results(output_file):
    """
    Load MIH results from the output HDF5 file.
    This function expects two datasets under the 'refs' group:
      - '<prefix>.res' : k-NN results (each row is a list of dataset indices)
      - '<prefix>.nres': range search results (each row is a list of indices within the given range)
    If a row in the range search result is padded (e.g. with -1), these entries are removed.
    """
    with h5py.File(output_file, "r") as f:
        if "refs" not in f:
            raise KeyError("No 'refs' key found in HDF5 file.")
        
        # Find MIH keys (assumes keys like 'mih0.res', 'mih0.nres', etc.)
        mih_keys = [key for key in f["refs"].keys() 
                    if key.startswith("mih") and key[3:].split('.')[0].isdigit()]
        if not mih_keys:
            raise KeyError("No valid MIH result keys found in HDF5 file.")
        
        # Select the MIH result with the highest index
        mih_index = max(int(k[3:].split('.')[0]) for k in mih_keys)
        mih_prefix = f"mih{mih_index}"
        
        res_key = f"{mih_prefix}.res"
        nres_key = f"{mih_prefix}.nres"
        
        if res_key not in f["refs"] or nres_key not in f["refs"]:
            raise KeyError(f"Expected datasets not found: {res_key} or {nres_key}")
        
        # Load k-NN results (adjusting indices if necessary)
        knn_results = [set(np.asarray(row, dtype=np.int64) - 1) 
                       for row in f["refs"][res_key][()]]
        
        # Load range search results as sets (remove any padding, e.g. -1)
        range_results = []
        for row in f["refs"][nres_key][()]:
            row_array = np.asarray(row, dtype=np.int64)
            valid = row_array[row_array != -1]  # Remove padding if used
            range_results.append(set(valid))
            
    return knn_results, range_results


if mih_success:
    try:
        knn_mih, range_mih = load_mih_results(mih_output_file)
        
        # Compute k-NN using SciPy (unchanged)
        def compute_knn(dataset, queries, K):
            knn_results = []
            for q in queries:
                dists = np.array([hamming(q, d) * B for d in dataset])
                knn_indices = set(np.argsort(dists)[:K])
                knn_results.append(knn_indices)
            return knn_results
        
        knn_scipy = compute_knn(dataset, queries, K)
        
        # Compute range search using the given fixed range threshold
        def compute_range_search(dataset, queries, r):
            range_results = []
            for q in queries:
                dists = np.array([hamming(q, d) * B for d in dataset])
                indices = set(np.where(dists < r)[0])
                range_results.append(indices)
            return range_results

        range_scipy = compute_range_search(dataset, queries, range_threshold)

        knn_match = all(knn_mih[i] == knn_scipy[i] for i in range(NQ))
        range_match = all(range_mih[i] == range_scipy[i] for i in range(NQ))

        print(f"k-NN results match: {knn_match}")
        print(f"Range search results match: {range_match}")

        if not knn_match:
            print("Differences in k-NN results!")
            print("MIH results:", knn_mih[:5])
            print("SciPy results:", knn_scipy[:5])

        if not range_match:
            print("Differences in range search results!")
            print("MIH results:", list(range_mih[:5]))
            print("SciPy results:", list(range_scipy[:5]))

    except KeyError as e:
        print(f"Error loading MIH results: {e}")
        print("Make sure MIH successfully wrote the expected datasets to the output file.")
else:
    print("MIH execution failed. Check the command and dataset format.")
