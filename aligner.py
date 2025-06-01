import os
import re
import json
import uuid
import torch
import random
import shutil
import pathlib
import pdfplumber
import tempfile
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

from config import AlignerConfig

cfg = AlignerConfig()

alignment_model, alignment_tokenizer = load_alignment_model(
    model_path=cfg.aligner_model,
    device=cfg.device,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils

_SPEAKER_TAG = re.compile(r'^[A-Z][A-Z0-9\s\'\-.]*\d?\s*:\s*')


def extract_relevant_text(pdf_path: str | pathlib.Path,
                          header_height: int = 70,
                          footer_height: int = 60,
                          gutter_width: int = 55) -> str:
    """Extract clean transcript lines from a TERES-style Supreme Court PDF."""
    spoken = []

    with pdfplumber.open(pdf_path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            if n == 1:
                continue  # Skip cover page

            x0, y0 = gutter_width, header_height
            x1, y1 = page.width, page.height - footer_height
            body = page.within_bbox((x0, y0, x1, y1))

            for raw_line in (body.extract_text() or "").splitlines():
                line = raw_line.strip()
                if not line or line.isdigit():
                    continue
                line = _SPEAKER_TAG.sub("", line).strip()
                if not line:
                    continue
                line = re.sub(r'\s{2,}', ' ', line)
                spoken.append(line)

    return '\n'.join(spoken)


def remove_ist_timestamps(text):
    pattern = r'^\s*\d{1,2}:\d{2}\s(?:AM|PM)\sIST\s*$'
    return '\n'.join(line for line in text.splitlines() if not re.match(pattern, line.strip()))


def remove_long_silences(audio, sample_rate, threshold=2):
    speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=sample_rate)
    if not speech_timestamps:
        return audio

    chunks = []
    last_end = 0

    for chunk in speech_timestamps:
        start, end = chunk['start'], chunk['end']
        if start > last_end:
            silence_duration = (start - last_end) / sample_rate
            if silence_duration < threshold:
                chunks.append(audio[last_end:start])
        chunks.append(audio[start:end])
        last_end = end

    if last_end < len(audio):
        trailing_silence = audio[last_end:]
        if (len(audio) - last_end) / sample_rate < threshold:
            chunks.append(trailing_silence)

    return torch.cat(chunks) if chunks else audio


def process_audio_file(audio_file):
    """Aligns and segments one audio file based on sentence-level timestamps."""
    os.makedirs("final_data", exist_ok=True)

    audio_file_path = os.path.join(cfg.audio_dir, audio_file)
    wav = read_audio(audio_file_path, sampling_rate=16000)
    sf_audio = remove_long_silences(wav, 16000)
    print("long silences removed")

    uid = uuid.uuid4()
    processed_audio_path = f"processed_{uid}_{audio_file}"
    sf.write(processed_audio_path, sf_audio, 16000)

    audio_waveform = load_audio(processed_audio_path, alignment_model.dtype, alignment_model.device)

    transcript_path = os.path.join(cfg.transcript_dir, audio_file.replace('.wav', '.pdf').replace('audio_', 'transcript_'))
    transcript_text = extract_relevant_text(transcript_path)
    transcript_text = remove_ist_timestamps(transcript_text)

    with open("tmp.txt", "w") as f:
        f.write(transcript_text)

    tokens_starred, text_starred = preprocess_text(
        transcript_text,
        romanize=True,
        language=cfg.language,
        split_size=cfg.split_size,
    )

    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=cfg.batch_size)
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    sentence_timestamps = postprocess_results(text_starred, spans, stride, scores)

    chunk_index = 0
    i = 0
    total_segments = len(sentence_timestamps)

    while i < total_segments:
        target_duration = random.uniform(5, 30)
        start_time = sentence_timestamps[i]['start']
        texts = [sentence_timestamps[i]['text']]
        end_time = sentence_timestamps[i]['end']
        i += 1

        while i < total_segments and (sentence_timestamps[i]['end'] - start_time) <= 30:
            if sentence_timestamps[i]['end'] - start_time > target_duration:
                break
            texts.append(sentence_timestamps[i]['text'])
            end_time = sentence_timestamps[i]['end']
            i += 1

        duration = end_time - start_time
        if duration < 5:
            continue

        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        chunk_audio = sf_audio[start_sample:end_sample]
        chunk_text = " ".join(texts).replace("\n", " ").strip()

        base_name = os.path.splitext(audio_file)[0]
        chunk_id = f"{base_name}_{chunk_index:03d}"
        sf.write(f"final_data/{chunk_id}.wav", chunk_audio, 16000)
        with open(f"final_data/{chunk_id}.txt", "w") as f:
            f.write(chunk_text)

        chunk_index += 1


def main():
    for audio_file in os.listdir(cfg.audio_dir):
        if audio_file.endswith('.wav'):
            print(audio_file)
            process_audio_file(audio_file)

if __name__ == "__main__":
    main()
