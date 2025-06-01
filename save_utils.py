# save_utils.py
import os
import numpy as np
from pathlib import Path

def sanitize_filename(text):
    return text.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")

def save_keypoints(keypoints, sample, output_dir):
    label = sample["label"]
    text = sanitize_filename(sample["text"])
    file_id = sanitize_filename(sample["file"])
    fname = f"{label}__{text}__{file_id}.npz"
    path = Path(output_dir) / fname

    if keypoints.shape == ():  # skip saving empty keypoints
        print(f"[Skip] Empty keypoints for {fname}, not saving.")
        return

    np.savez_compressed(
        path,
        keypoints=keypoints,
        label=label,
        text=text
    )
    print(f"[Saved] {fname} -- shape {keypoints.shape}")
