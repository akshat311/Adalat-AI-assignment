import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio, Features, Value
from huggingface_hub import HfApi, login

FINAL_DATA_DIR = "final_data"
EXCEL_PATH = "SC Transcripts _ ML Assignment Speech.xlsx"
OUTPUT_DIR = "commonvoice_export"
SPLIT_RATIO = 0.9
REPO_NAME = "supreme-court-stt"
HF_USERNAME = "akshat311"

def load_metadata():
    """Load case metadata from the Excel sheet."""
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df = df[['Case Name', 'Case Number', 'Hearing Date']]
    df.index = df.index + 1  # Match Excel 1-based indexing
    return df.to_dict(orient='index')

def collect_entries(metadata_dict):
    """Scan the final_data directory for .wav and .txt pairs and enrich with metadata."""
    entries = []

    for file in tqdm(os.listdir(FINAL_DATA_DIR), desc="Collecting entries"):
        if not file.endswith('.wav'):
            continue

        audio_path = os.path.join(FINAL_DATA_DIR, file)
        base_name = file.replace('.wav', '')
        txt_path = os.path.join(FINAL_DATA_DIR, base_name + '.txt')

        if not os.path.exists(txt_path):
            continue

        try:
            video_num_str, chunk_num = base_name.replace("audio_", "").split("_")
            video_num = int(video_num_str)
        except ValueError:
            continue  # skip improperly named files

        transcript = open(txt_path, "r", encoding='utf-8').read().strip()
        meta = metadata_dict.get(video_num + 1, {})

        entries.append({
            "path": os.path.relpath(audio_path, start=OUTPUT_DIR),
            "audio": audio_path,
            "sentence": transcript,
            "video_id": video_num,
            "chunk_id": int(chunk_num),
            "case_name": str(meta.get("Case Name", "") or ""),
            "case_number": str(meta.get("Case Number", "") or ""),
            "hearing_date": str(meta.get("Hearing Date", "") or ""),
        })

    return entries

def save_json(entries):
    """Shuffle and split the dataset into train/test JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.shuffle(entries)
    split_idx = int(len(entries) * SPLIT_RATIO)

    train_set = entries[:split_idx]
    test_set = entries[split_idx:]

    with open(os.path.join(OUTPUT_DIR, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "test.json"), "w", encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Train: {len(train_set)} samples, Test: {len(test_set)} samples.")
    print(f"📂 Output saved in '{OUTPUT_DIR}/train.json' and 'test.json'")
    return train_set, test_set

def push_to_hub(train_data, test_data, repo_id):
    """Upload the dataset to Hugging Face Hub."""
    print(f"🚀 Uploading dataset to Hugging Face Hub: {repo_id}")

    features = Features({
        "path": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "sentence": Value("string"),
        "video_id": Value("int32"),
        "chunk_id": Value("int32"),
        "case_name": Value("string"),
        "case_number": Value("string"),
        "hearing_date": Value("string"),
    })

    train_ds = Dataset.from_list(train_data, features=features)
    test_ds = Dataset.from_list(test_data, features=features)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    dataset_dict.push_to_hub(repo_id)
    print(f"✅ Dataset pushed to https://huggingface.co/datasets/{repo_id}")

def main(push_to_hub_flag):
    metadata_dict = load_metadata()
    entries = collect_entries(metadata_dict)
    train_data, test_data = save_json(entries)

    if push_to_hub_flag:
        repo_id = f"{HF_USERNAME}/{REPO_NAME}"
        login()  # will prompt for HF token
        push_to_hub(train_data, test_data, repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and optionally upload CommonVoice-formatted dataset")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to Hugging Face Hub")
    args = parser.parse_args()
    main(args.push_to_hub)
