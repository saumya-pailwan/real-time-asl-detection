import json
import numpy as np
import os

# Paths to your JSON splits (adjust if needed)
TRAIN_JSON = "train_keypoints_100_max_frame.json"
VAL_JSON   = "val_keypoints_100_max_frame.json"
TEST_JSON  = "test_keypoints_100_max_frame.json"

MAX_FRAMES     = 100
HAND_LANDMARKS = 21
FEATURE_DIM    = HAND_LANDMARKS * 2 * 3  # 21 points × 2 hands × (x,y,z)

def load_and_tabulate_with_mask(json_path):
    """
    Reads a split JSON, skips entries with processed_frames == 0, and returns:
      - X:    np.float32 array of shape (n_samples, MAX_FRAMES, FEATURE_DIM)
      - masks: np.uint8 array of shape (n_samples, MAX_FRAMES), with 1=real frame, 0=padded
      - labels: list of string labels, length = n_samples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_list    = []
    mask_list = []
    labels    = []

    for entry in data:
        keypoints_data   = entry.get("keypoints_data", {})
        video_info       = keypoints_data.get("video_info", {})
        processed_frames = video_info.get("processed_frames", 0)

        # Skip any sample with zero processed frames
        if processed_frames == 0:
            continue

        # We'll fill up to MAX_FRAMES rows of 126 features
        sample_matrix = np.zeros((MAX_FRAMES, FEATURE_DIM), dtype=np.float32)
        # Mask = 1 for real frames, 0 for padding
        sample_mask   = np.zeros((MAX_FRAMES,), dtype=np.uint8)

        frames = keypoints_data.get("keypoints", [])
        # How many actual frames do we have?  (cap it at MAX_FRAMES)
        num_real = min(len(frames), MAX_FRAMES)

        for i in range(num_real):
            feature_vec = np.zeros((FEATURE_DIM,), dtype=np.float32)

            frame_info = frames[i]

            # LEFT HAND (21 points × 3)
            left = frame_info.get("left_hand")
            if left:
                for idx, (x, y, z) in enumerate(left):
                    base = idx * 3
                    feature_vec[base : base+3] = [x, y, z]
            # If left is None, those 63 slots stay at zero

            # RIGHT HAND (21 points × 3)
            right = frame_info.get("right_hand")
            if right:
                for idx, (x, y, z) in enumerate(right):
                    base = (HAND_LANDMARKS * 3) + idx * 3
                    feature_vec[base : base+3] = [x, y, z]
            # If right is None, those 63 slots stay at zero

            sample_matrix[i] = feature_vec
            sample_mask[i]   = 1  # mark this row as “real”

        # Remaining rows [num_real:MAX_FRAMES] stay zero and mask stays 0

        X_list.append(sample_matrix)
        mask_list.append(sample_mask)
        labels.append(entry["original_data"]["clean_text"])

    X_array    = np.stack(X_list, axis=0)    # shape = (n_samples, MAX_FRAMES, FEATURE_DIM)
    mask_array = np.stack(mask_list, axis=0) # shape = (n_samples, MAX_FRAMES)
    return X_array, mask_array, labels

if __name__ == "__main__":
    os.makedirs("processed_data", exist_ok=True)

    # Build train/val/test splits with masks
    X_train, mask_train, y_train = load_and_tabulate_with_mask(TRAIN_JSON)
    X_val,   mask_val,   y_val   = load_and_tabulate_with_mask(VAL_JSON)
    X_test,  mask_test,  y_test  = load_and_tabulate_with_mask(TEST_JSON)

    # Save all three arrays per split into a single .npz
    np.savez_compressed("processed_data/train_100_max_frame.npz",
                        X=X_train,
                        mask=mask_train,
                        y=y_train)
    np.savez_compressed("processed_data/val_100_max_frame.npz",
                        X=X_val,
                        mask=mask_val,
                        y=y_val)
    np.savez_compressed("processed_data/test_100_max_frame.npz",
                        X=X_test,
                        mask=mask_test,
                        y=y_test)

    print("Saved:")
    print(f"  train.npz → X: {X_train.shape}, mask: {mask_train.shape}, labels: {len(y_train)}")
    print(f"   val.npz → X: {X_val.shape},   mask: {mask_val.shape},   labels: {len(y_val)}")
    print(f"  test.npz → X: {X_test.shape},  mask: {mask_test.shape},  labels: {len(y_test)}")