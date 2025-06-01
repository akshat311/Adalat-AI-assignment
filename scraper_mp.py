import pandas as pd
import requests
import os
from PyPDF2 import PdfReader
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
excel_file_path = 'SC Transcripts _ ML Assignment Speech.xlsx'
os.makedirs('audio_files', exist_ok=True)
os.makedirs('transcripts', exist_ok=True)

# Load Excel
excel_data = pd.read_excel(excel_file_path)

# Function to download audio from YouTube and convert to WAV
def download_audio_youtube(youtube_url, save_path):
    command = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '--output', save_path,
        '--postprocessor-args', "-ar 16000 -ac 1",
        youtube_url,
    ]
    subprocess.run(command, check=True)



# Process a single row (used by multiprocessing)
def process_row(args):
    index, row = args
    youtube_url = row['Oral Hearing Link']
    transcript_url = row['Transcript Link']
    
    try:
        # Audio
        audio_file_path = f"audio_files/audio_{index}.wav"
        download_audio_youtube(youtube_url, audio_file_path)
        
        # Transcript
        response = requests.get(transcript_url)
        with open(f"transcripts/transcript_{index}.pdf", "wb") as f:
            f.write(response.content)
        return f"✅ Completed index {index}"
    except Exception as e:
        return f"❌ Error at index {index}: {e}"

# Main
if __name__ == "__main__":
    args_list = list(excel_data.iterrows())

    with Pool(processes=1) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, args_list), total=len(args_list)))

    # Print summary
    for r in results:
        print(r)
