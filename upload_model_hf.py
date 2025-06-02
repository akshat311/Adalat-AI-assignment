#!/usr/bin/env python

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from transformers import WhisperProcessor

REQUIRED_FILES = {
    "model.safetensors",
    "config.json",
    "generation_config.json"
}

# This is for processor-related files
PROCESSOR_FILES = {
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "feature_extractor.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}

def ensure_login(token: str | None):
    if token:
        HfFolder.save_token(token)
    elif HfFolder.get_token() is None:
        env_tok = os.getenv("HF_TOKEN")
        if env_tok:
            HfFolder.save_token(env_tok)
        else:
            raise RuntimeError("Please login with `huggingface-cli login` or provide a token.")


def main(args):
    ensure_login(args.token)
    api = HfApi()

    model_dir = Path(args.model_dir)
    parent_dir = model_dir.parent

    # Create repo if not exists
    if not api.repo_exists(args.repo_id):
        create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
        print(f"🆕 Created repo: {args.repo_id}")
    else:
        print(f"↻ Repo already exists: {args.repo_id}")

    # Upload only essential model files
    print("⬆️ Uploading model files...")
    for file_name in REQUIRED_FILES:
        file_path = model_dir / file_name
        if file_path.exists():
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Add {file_name}",
            )
        else:
            print(f"⚠️  Missing expected file: {file_name}")

    # Save & upload processor from parent dir
    print("⬆️ Uploading processor files...")
    tmp_proc_dir = Path("tmp_proc_dir")
    if tmp_proc_dir.exists():
        shutil.rmtree(tmp_proc_dir)
    tmp_proc_dir.mkdir()

    processor = WhisperProcessor.from_pretrained(parent_dir)
    processor.save_pretrained(tmp_proc_dir)

    for file_name in PROCESSOR_FILES:
        file_path = tmp_proc_dir / file_name
        if file_path.exists():
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Add processor file: {file_name}",
            )

    shutil.rmtree(tmp_proc_dir)
    print(f"✅ Model is live → https://huggingface.co/{args.repo_id}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=False,
        default = "/root/personal/Adalat-AI-assignment/train_medium/checkpoint-900",
        help="Directory that contains the fine-tuned checkpoint",
    )
    parser.add_argument(
        "--repo_id",
        required=False,
        default = "akshat311/legal-whisper-medium",
        help="Target repo on the Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push to a private repo instead of public",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token with write access (or rely on cached login / $HF_TOKEN)",
    )
    args = parser.parse_args()
    main(args)
