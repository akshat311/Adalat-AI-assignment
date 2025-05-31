import os
import re
import random
import soundfile as sf
import numpy as np
from collections import defaultdict

CHUNKS_DIR = "chunks"
MERGED_DIR = "merged_chunks"
os.makedirs(MERGED_DIR, exist_ok=True)

def extract_base_and_start(filename):
    """Extract original filename and start time"""
    match = re.match(r"(.+?\.wav)_(\d+)_(\d+)\.wav", filename)
    if match:
        base = match.group(1)
        start = int(match.group(2))
        return base, start
    return None, None

def load_chunks():
    """Load and group chunks by base audio name"""
    groups = defaultdict(list)
    for file in os.listdir(CHUNKS_DIR):
        if not file.endswith(".wav"):
            continue
        base, start = extract_base_and_start(file)
        print(base, start)
        if base is None:
            continue
        wav_path = os.path.join(CHUNKS_DIR, file)
        print(wav_path)
        txt_path = wav_path.rsplit('.wav', 1)[0] + '.txt'
        print(txt_path)
        if not os.path.exists(txt_path):
            continue
        audio, sr = sf.read(wav_path)
        duration = len(audio) / sr
        groups[base].append({
            "audio_path": wav_path,
            "text_path": txt_path,
            "text": open(txt_path).read().strip(),
            "duration": duration,
            "start": start,
            "sr": sr
        })
    # Sort each group by start time
    for base in groups:
        groups[base] = sorted(groups[base], key=lambda x: x["start"])
    return groups

def merge_sequential_chunks(groups):
    idx = 0
    for base, chunks in groups.items():
        i = 0
        while i < len(chunks):
            target_duration = random.uniform(5, 30)
            total_duration = 0
            audio_segments = []
            text_segments = []

            while i < len(chunks) and total_duration + chunks[i]['duration'] <= target_duration:
                chunk = chunks[i]
                audio, _ = sf.read(chunk['audio_path'])
                audio_segments.append(audio)
                text_segments.append(chunk['text'])
                total_duration += chunk['duration']
                i += 1

            # Handle too-long individual chunk
            if not audio_segments and i < len(chunks):
                chunk = chunks[i]
                audio, _ = sf.read(chunk['audio_path'])
                audio_segments.append(audio)
                text_segments.append(chunk['text'])
                i += 1

            merged_audio = np.concatenate(audio_segments)
            merged_text = " ".join(text_segments)

            merged_base = f"{idx:04d}"
            sf.write(f"{MERGED_DIR}/{merged_base}.wav", merged_audio, chunks[0]['sr'])
            with open(f"{MERGED_DIR}/{merged_base}.txt", "w") as f:
                f.write(merged_text)

            idx += 1

def main():
    groups = load_chunks()
    print(f"Found {len(groups)} original audio files.")
    merge_sequential_chunks(groups)
    print(f"Sequential merged chunks saved to '{MERGED_DIR}'.")

if __name__ == "__main__":
    main()
