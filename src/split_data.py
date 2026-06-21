import os
import numpy as np

def process_and_split_smd():
    raw_dir = "data/raw"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load telemetry metrics
    train_base = np.loadtxt(os.path.join(raw_dir, "machine-1-1.txt"), delimiter=',')
    test_live = np.loadtxt(os.path.join(raw_dir, "machine-1-1_test.txt"), delimiter=',')
    test_labels = np.loadtxt(os.path.join(raw_dir, "machine-1-1_labels.txt"), delimiter=',')
    
    # 2. Run the 70/30 split logic on clean normal logs
    split_idx = int(len(train_base) * 0.7)
    train_split = train_base[:split_idx]
    val_split = train_base[split_idx:]
    
    # 3. Write data to the processed staging directory
    np.savetxt(os.path.join(out_dir, "train.csv"), train_split, delimiter=",")
    np.savetxt(os.path.join(out_dir, "validation.csv"), val_split, delimiter=",")
    np.savetxt(os.path.join(out_dir, "demo_stream.csv"), test_live, delimiter=",")
    np.savetxt(os.path.join(out_dir, "demo_labels.csv"), test_labels, delimiter=",")
    
    print(f"Split complete! Train: {train_split.shape}, Val: {val_split.shape}, Demo: {test_live.shape}")

if __name__ == "__main__":
    process_and_split_smd()