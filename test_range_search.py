import numpy as np
import h5py
import os
from scipy.spatial.distance import hamming

# Parameters
N = 1000      # Number of dataset binary codes
NQ = 10       # Number of query points
B = 64        # Number of bits per code
K = 5         # Number of nearest neighbors
m = 4         # Number of hash tables for MIH

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

# Run MIH with k-NN and range search
mih_command = f"./build/mih {dataset_file} {mih_output_file} -N {N} -B {B} -m {m} -Q {NQ} -K {K} -r"
mih_success = os.system(mih_command) == 0

# Load MIH results if successful
def load_mih_results(output_file):
    with h5py.File(output_file, "r") as f:
        print("Available keys in HDF5 file:", list(f.keys()))  # Debugging
        if "refs" not in f:
            raise KeyError("No 'refs' key found in HDF5 file.")
        mih_keys = [key for key in f["refs"].keys() if key.startswith("mih") and key[3:].split('.')[0].isdigit()]
        if not mih_keys:
            raise KeyError("No valid MIH result keys found in HDF5 file.")
        mih_path = max(mih_keys, key=lambda x: int(x[3:].split('.')[0]))
        
        # Ensure correct dataset paths
        res_path = f"/refs/{mih_path}.res"
        nres_path = f"/refs/{mih_path}.nres"
        
        if res_path not in f or nres_path not in f:
            raise KeyError(f"Expected datasets not found: {res_path} or {nres_path}")
        
        knn_results = [set(row) for row in np.array(f[res_path])]
        range_counts = np.array(f[nres_path])[:, 0]  # Use only the first column
    return knn_results, range_counts

if mih_success:
    try:
        knn_mih, range_mih = load_mih_results(mih_output_file)
        
        # Compute k-NN and range search using SciPy
        def compute_knn_and_range(dataset, queries, K):
            knn_results = []
            range_counts = []
            for q in queries:
                dists = np.array([hamming(q, d) * B for d in dataset])  # Scale to bit count
                knn_indices = set(np.argsort(dists)[:K])
                knn_results.append(knn_indices)
                max_hamming = max(dists[list(knn_indices)])
                range_count = np.sum(dists <= max_hamming)
                range_counts.append(range_count)
            return knn_results, np.array(range_counts)

        knn_scipy, range_scipy = compute_knn_and_range(dataset, queries, K)

        knn_match = all(knn_mih[i] == knn_scipy[i] for i in range(NQ))
        range_match = np.all(range_mih == range_scipy)

        print(f"k-NN results match: {knn_match}")
        print(f"Range search results match: {range_match}")

        if not knn_match:
            print("Differences in k-NN results!")
            print("MIH results:", knn_mih[:5])
            print("SciPy results:", knn_scipy[:5])

        if not range_match:
            print("Differences in range search results!")
            print("MIH results:", range_mih[:5])
            print("SciPy results:", range_scipy[:5])

    except KeyError as e:
        print(f"Error loading MIH results: {e}")
        print("Make sure MIH successfully wrote the expected datasets to the output file.")
else:
    print("MIH execution failed. Check the command and dataset format.")
