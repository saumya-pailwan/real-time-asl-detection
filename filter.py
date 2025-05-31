import json
import os

SPLIT_PATHS = {
    "train": "MS-ASL/MSASL_train.json",
    "val":   "MS-ASL/MSASL_val.json",
    "test":  "MS-ASL/MSASL_test.json"
}

top100_signs = ['about', 'again', 'ask', 'bad', 'boy', 'but', 'buy', 'can', 'come',
 'different', 'drink', 'easy', 'eat', 'family', 'feel', 'few', 'find',
 'fine', 'finish', 'for', 'forget', 'friend', 'get', 'girl', 'give', 'go',
 'good', 'happy', 'hard', 'have', 'he', 'hello', 'help', 'home', 'how',
 'know', 'later', 'like', 'little', 'live', 'look', 'make', 'many', 'me',
 'meet', 'more', 'my', 'name', 'need', 'new', 'no', 'not', 'now', 'ok',
 'old', 'other', 'play', 'please', 'remember', 'right', 'sad', 'same',
 'say', 'school', 'see', 'she', 'sign', 'slow', 'some', 'sorry', 'stay',
 'take', 'talk', 'tell', 'thank you', 'their', 'they', 'thing', 'think',
 'time', 'tired', 'try', 'understand', 'use', 'wait', 'want', 'what',
 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'work', 'write',
 'wrong', 'yes', 'you', 'your']

FILTERED_DIR = "msasl_top100_splits"
os.makedirs(FILTERED_DIR, exist_ok=True)

all_top100_entries = []

#Filter out top 100 words from each MSASL splits
for split_name, path in SPLIT_PATHS.items():
	with open(path, "r", encoding="utf-8") as f:entries = json.load(f)

	filtered = []
	for entry in entries:
		token = entry["clean_text"].strip().lower()
		if token in top100_signs:
			filtered.append(entry)
	
	out_path = os.path.join(FILTERED_DIR, f"{split_name}_top100.json")
	with open(out_path, "w", encoding="utf-8") as fout:
		json.dump(filtered, fout, indent=2)
	print(f"→ {split_name} → {len(filtered)} entries saved to {out_path}")

#Split Verification to check for missing labels
split_paths = {
    "train": "msasl_top100_splits/train_top100.json",
    "val":   "msasl_top100_splits/val_top100.json",
    "test":  "msasl_top100_splits/test_top100.json"
}

all_present = set()

for split_name, path in split_paths.items():
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    labels = {entry["clean_text"].strip().lower() for entry in entries}
    print(f"{split_name:5s} split: {len(entries)} entries, {len(labels)} unique labels")
    all_present.update(labels)

print(f"\nCombined unique labels found across all splits ({len(all_present)} total):")
print(sorted(all_present))

missing = set(top100_signs) - all_present
if missing:
    print(f"\n--- MISSING TOKENS ({len(missing)}) ---")
    for w in sorted(missing):
        print(f"  • {w}")
else:
    print("\nEverything matched! No missing tokens from your 100-word list.")