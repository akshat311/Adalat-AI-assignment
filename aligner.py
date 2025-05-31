import os
import random
import uuid
import torch
import torchaudio
import soundfile as sf
from silero_vad import get_speech_timestamps, read_audio
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
import re

# Constants
LANGUAGE = "iso"  # ISO-639-3 Language code
DEVICE = "mps" if torch.mps.is_available() else "cpu"
BATCH_SIZE = 16
AUDIO_DIR = 'audio_files'
TRANSCRIPT_DIR = 'transcripts'

# Load the alignment model
alignment_model, alignment_tokenizer = load_alignment_model(
    model_path="facebook/wav2vec2-large-960h",
    device=DEVICE,
    dtype=torch.float16 if DEVICE == "mps" else torch.float32,
)

# Load model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def extract_relevant_text(text):
    """Extracts and cleans relevant text from the transcript."""
    text = re.sub(r'Transcribed\s+by\s+TERES', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)\s+\d+\s+(\w+)\b', r'\1 \2', text)
    text = re.sub(r'\d{1,2}:\d{2}\s+(AM|PM)\s+IST', '', text, flags=re.IGNORECASE)
    text = re.sub(r'END OF THIS PROCEEDING', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    pattern = re.compile(r'([A-Z\s\.\'\d\-]+):\s*(.*?)(?=\s+[A-Z\s\.\'\d\-]+:|$)')
    transcripts = []
    for match in pattern.finditer(text):
        speech = match.group(2)
        speech = re.sub(r'\s*\b\d+\b\s*', ' ', speech)
        transcripts.append(speech.strip())
    return '\n'.join(transcripts)

def remove_long_silences(audio, sample_rate, threshold=2):
    """Removes long silences from the audio based on VAD results."""
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    if not speech_timestamps:
        return audio  # No speech detected
    chunks = []
    last_end = 0
    for i, chunk in enumerate(speech_timestamps):
        start, end = chunk['start'], chunk['end']
        if start > last_end:
            silence_duration = (start - last_end) / sample_rate
            if silence_duration < threshold:
                chunks.append(audio[last_end:start])
        chunks.append(audio[start:end])
        last_end = end
    if last_end < len(audio):
        trailing_silence = audio[last_end:]
        silence_duration = (len(audio) - last_end) / sample_rate
        if silence_duration < threshold:
            chunks.append(trailing_silence)
    return torch.cat(chunks) if chunks else audio

def process_audio_file(audio_file):
    """Processes a single audio file and saves natural segments using sentence-level alignments."""
    os.makedirs("final_data", exist_ok=True)
    
    audio_file_path = os.path.join(AUDIO_DIR, audio_file)
    wav = read_audio(audio_file_path, sampling_rate=16000)
    sf_audio = remove_long_silences(wav, 16000)
    
    uid = uuid.uuid4()
    processed_audio_path = f"processed_{uid}_{audio_file}"
    sf.write(processed_audio_path, sf_audio, 16000)
    
    audio_waveform = load_audio(processed_audio_path, alignment_model.dtype, alignment_model.device)
    
    transcript_file_path = os.path.join(TRANSCRIPT_DIR, audio_file.replace('.wav', '.txt').replace('audio_', 'transcript_'))
    with open(transcript_file_path, "r") as f:
        transcript_text = f.read().replace("\n", " ").strip()

    transcript_text = extract_relevant_text(transcript_text)
    with open("tmp.txt", "w") as f:
        f.write(transcript_text)
    tokens_starred, text_starred = preprocess_text(
        transcript_text,
        romanize=True,
        language=LANGUAGE,
        split_size="sentence",
    )

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )
    
    spans = get_spans(tokens_starred, segments, blank_token)
    sentence_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # Now segment audio based on sentence timestamps with 5–30 sec blocks
    chunk_index = 0
    i = 0
    total_segments = len(sentence_timestamps)
    while i < total_segments:
        target_duration = random.uniform(5, 30)
        start_time = sentence_timestamps[i]['start']
        texts = [sentence_timestamps[i]['text']]
        end_time = sentence_timestamps[i]['end']
        i += 1

        # Accumulate until the next sentence would exceed the target or we hit the limit
        while i < total_segments and (sentence_timestamps[i]['end'] - start_time) <= 30:
            proposed_end = sentence_timestamps[i]['end']
            if proposed_end - start_time > target_duration:
                break
            texts.append(sentence_timestamps[i]['text'])
            end_time = proposed_end
            i += 1

        duration = end_time - start_time
        if duration < 5:
            continue  # skip short segments

        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        chunk_audio = sf_audio[start_sample:end_sample]
        chunk_text = " ".join(texts)

        base_name = os.path.splitext(audio_file)[0]
        chunk_id = f"{base_name}_{chunk_index:03d}"
        sf.write(f"final_data/{chunk_id}.wav", chunk_audio, 16000)
        with open(f"final_data/{chunk_id}.txt", "w") as f:
            f.write(chunk_text)
        chunk_index += 1

def main():
    """Main function to process all audio files in the directory."""
    for audio_file in os.listdir(AUDIO_DIR):
        if audio_file.endswith('.wav'):
            process_audio_file(audio_file)

if __name__ == "__main__":
    main()
