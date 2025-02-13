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

# Save in HDF5 format (Ensuring Correct Dataset Names)
dataset_file = "dataset_test.h5"
queries_file = "queries_test.h5"

with h5py.File(dataset_file, "w") as f:
    f.create_dataset("B", data=dataset_packed)

with h5py.File(queries_file, "w") as f:
    f.create_dataset("Q", data=queries_packed)  # Ensure correct name

# Run MIH with k-NN and range search
mih_output_file = "mih_results.h5"
mih_command = f"./build/mih {dataset_file} {mih_output_file} -N {N} -B {B} -m {m} -Q {NQ} -K {K} -r"

mih_success = os.system(mih_command) == 0

# Load MIH results if successful
def load_mih_results(output_file):
    """Parses MIH output stored in HDF5 and dynamically finds dataset keys."""
    with h5py.File(output_file, "r") as f:
        print("Available keys in HDF5 file:", list(f.keys()))  # Debugging
        
        # Identify correct keys for MIH results
        mih_keys = [key for key in f["refs"].keys() if key.startswith("mih")]
        if not mih_keys:
            raise KeyError("No valid MIH keys found in output file.")
        mih_path = max(mih_keys, key=lambda x: int(x[3:]))
        mih_path = f"/refs/{mih_path}"

        # Extract k-NN and range search results
        knn_results = np.array(f[f"{mih_path}.res"])
        range_counts = np.array(f[f"{mih_path}.nres"])[:, 0]  # Extract first column only
    
    return knn_results, range_counts

if mih_success:
    try:
        knn_mih, range_mih = load_mih_results(mih_output_file)
        
        # Compute k-NN and range search using SciPy
        def compute_knn_and_range(dataset, queries, K):
            """Computes k-NN and range search counts using SciPy Hamming distance."""
            knn_results = []
            range_counts = []

            for q in queries:
                # Compute Hamming distances
                dists = np.array([hamming(q, d) * B for d in dataset])  # Scale to bit count
                # Find k-NN
                knn_indices = np.argsort(dists)[:K]
                knn_results.append(set(knn_indices))  # Store as a set for order-independent comparison
                # Compute max Hamming distance of k-NN
                max_hamming = dists[knn_indices[-1]]
                # Count all within range
                range_count = np.sum(dists <= max_hamming)
                range_counts.append(range_count)

            return np.array(knn_results), np.array(range_counts)

        knn_scipy, range_scipy = compute_knn_and_range(dataset, queries, K)

        # Compare results
        knn_match = all(knn_mih[i].tolist() == list(knn_scipy[i]) for i in range(NQ))
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
