# extract_and_save_keypoints.py
import os
import json
from pathlib import Path
from tqdm import tqdm
from stream_and_extract import process_video_clip
from save_utils import save_keypoints


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    samples = []
    for entry in data:
        if "url" not in entry or not entry["url"].startswith("http"):
            continue  # skip broken entries
        samples.append({
            "url": entry["url"],
            "label": entry["label"],
            "text": entry["clean_text"],
            "start_time": entry["start_time"],
            "end_time": entry["end_time"],
            "file": entry["file"]
        })
    return samples


if __name__ == "__main__":
    json_path = "MS-ASL/MSASL_train.json"
    output_dir = Path("data/keypoints/")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_json(json_path)

    for sample in tqdm(samples[:30]):  # we can use more later
        try:
            keypoints = process_video_clip(sample)
            save_keypoints(keypoints, sample, output_dir)
        except Exception as e:
            print(f"[Error] {sample['file']} - {e}")