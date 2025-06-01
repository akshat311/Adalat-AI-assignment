import pandas as pd
import requests
import os
from PyPDF2 import PdfReader
import subprocess
from tqdm import tqdm

# Define the path to the Excel file
excel_file_path = 'SC Transcripts _ ML Assignment Speech.xlsx'

# Read the Excel file
excel_data = pd.read_excel(excel_file_path)

# Create directories to store downloaded audio files and transcripts
os.makedirs('audio_files', exist_ok=True)
os.makedirs('transcripts', exist_ok=True)

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


# Iterate over each row in the Excel file
for index, row in tqdm(excel_data.iterrows()):
    youtube_url = row['Oral Hearing Link']
    transcript_url = row['Transcript Link']
    
    # Download the audio file using yt-dlp
    audio_file_path = f"audio_files/audio_{index}.wav"
    download_audio_youtube(youtube_url, audio_file_path)
    
    # Extract text from the PDF
    response = requests.get(transcript_url)
    with open(f"transcripts/transcript_{index}.pdf", "wb") as f:
        f.write(response.content)
    
    # Save the transcript text
    with open(f"transcripts/transcript_{index}.txt", "w") as f:
        f.write(transcript_text) 