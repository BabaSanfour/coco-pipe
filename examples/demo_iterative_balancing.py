
import numpy as np
import pandas as pd
from coco_pipe.io import DataContainer

def run_demo():
    print("Demo: Iterative Undersampling with Randomness")
    print("============================================")

    # 1. create imbalanced data: 600 of class 'A', 50 of class 'B'
    n_a = 600
    n_b = 50
    
    n_total = n_a + n_b
    X = np.random.randn(n_total, 2)
    y = np.array(['A']*n_a + ['B']*n_b)
    ids = np.arange(n_total)
    
    container = DataContainer(X=X, dims=('obs', 'feat'), y=y, ids=ids)
    
    print(f"Original Distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # 2. Run balance multiple times with different seeds
    seeds = [1, 2, 3]
    
    # Store indices selected for the majority class 'A' to compare overlaps
    selected_indices_a = []
    
    for seed in seeds:
        print(f"\n--- Balancing with random_state={seed} ---")
        balanced = container.balance(target='y', strategy='undersample', random_state=seed)
        
        # Check counts
        counts = pd.Series(balanced.y).value_counts().to_dict()
        print(f"Balanced Counts: {counts}")
        
        # Get indices of class A in this balanced set
        # We can find them by looking at IDs or matching logic, but here we can just inspect the subset
        # Since ids are just 0..649, let's see which of the first 600 were picked
        subset_ids = balanced.ids
        class_a_mask = balanced.y == 'A'
        ids_a = subset_ids[class_a_mask]
        
        print(f"Class A sample size: {len(ids_a)}")
        print(f"First 5 IDs selected for A: {ids_a[:5]}")
        
        selected_indices_a.append(set(ids_a))

    # 3. Analyze overlap
    set1, set2, set3 = selected_indices_a
    overlap_1_2 = len(set1.intersection(set2))
    overlap_1_3 = len(set1.intersection(set3))
    
    print("\n--- Overlap Analysis ---")
    print(f"Overlap between Run 1 and Run 2 (Class A): {overlap_1_2} / 50")
    print(f"Overlap between Run 1 and Run 3 (Class A): {overlap_1_3} / 50")
    
    if overlap_1_2 < 50 and overlap_1_3 < 50:
         print("\nSUCCESS: Different seeds produced different subsets!")
    else:
         print("\nFAILURE: Subsets are identical despite different seeds.")

if __name__ == "__main__":
    run_demo()
