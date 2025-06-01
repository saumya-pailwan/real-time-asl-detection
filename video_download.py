import os
import json
import subprocess
from urllib.parse import urlparse, parse_qs

SPLIT_JSONS = {
    "train": "msasl_top100_splits/train_top100.json",
    "val":   "msasl_top100_splits/val_top100.json",
    "test":  "msasl_top100_splits/test_top100.json"
}
CLIPS_DIR = "clips_top100_segments"
os.makedirs(CLIPS_DIR, exist_ok=True)

def youtube_id_from_url(url: str) -> str:
	# Parse url and return video id
	parsed = urlparse(url)
	if parsed.hostname in ("www.youtube.com", "youtube.com"):
		qs = parse_qs(parsed.query)
		return qs.get("v", [None])[0]
	else:
		return None

def seconds_to_hhmmss(sec: float) -> str:
	# Convert seconds to hh:mm:ss format
	hours = int(sec // 3600)
	minutes = int((sec % 3600) // 60)
	seconds = sec - (hours * 3600 + minutes * 60)
	return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def download_relevant_clip(url: str, start: float, end: float, output_path: str):
	# Download only relevant part of video
	start_hhmm = seconds_to_hhmmss(start)
	end_hhmm   = seconds_to_hhmmss(end)
	section_arg = f"*{start_hhmm}-{end_hhmm}"

	cmd = f"yt-dlp --quiet --format bestvideo[height<=144]+bestaudio/best[height<=144] --merge-output-format mp4 --download-sections {section_arg} --output {output_path} {url}"

	try:
		subprocess.run(cmd, check=True)
	except subprocess.CalledProcessError as e:
		print(f"[ERROR] yt-dlp failed for URL {url} with section {section_arg}")
		print("  → Command:", "".join(cmd))
		raise e
	
def clip_split(split_name: str, json_path: str):
	# Load JSON and create new directory
	with open(json_path, "r", encoding="utf-8") as f:
		entries = json.load(f)
	print(f"\n→ Processing split '{split_name}' ({len(entries)} entries)")
	
	split_folder = os.path.join(CLIPS_DIR, split_name)
	os.makedirs(split_name, exist_ok=True)

	# Loop over each entry in the split and get parameters
	for entry in entries:
		url = entry["url"]
		word = entry["clean_text"].strip().lower()
		start_s = float(entry["start_time"])
		end_s = float(entry["end_time"])
		signer = entry.get("signer_id", 0)
		frame_id = entry.get("start", 0)
		
		vid_id = youtube_id_from_url(url)
		if vid_id is None:
			print(f"[WARN] could not parse YouTube ID from URL: {url}")
			continue
		
		clip_filename = f"{word}_{vid_id}_{signer}_{start_s:.3f}_{end_s:.3f}.mp4"
		clip_path = os.path.join(split_folder, clip_filename)

		if os.path.exists(clip_path):
			continue
	
		print(f"[{split_name.upper()}:CLIP] {word:10s} | {vid_id} | "
			f"{start_s:.3f}-{end_s:.3f}s  →  {split_name}/{clip_filename}")
		
		# Download Clip from parameters and save
		try:
			download_relevant_clip(url, start_s, end_s, clip_path)
		except Exception:
			print(f"[FAIL] Could not clip segment for entry: {entry}")
			continue
		
	print(f"  → Done with '{split_name}' split. Clips saved to '{split_folder}'.\n")

def main():
	for split_name, json_path in SPLIT_JSONS.items():
		if not os.path.exists(json_path):
			print(f"[ERROR] Expected JSON not found: {json_path}. Skipping '{split_name}'.")
			continue
		clip_split(split_name, json_path)
	print("All splits processed. Check the directories under", CLIPS_DIR)

if __name__ == "__main__":
    main()