# 🎧 Adalat-AI ASR Pipeline

This project contains an end-to-end pipeline to fine-tune a Whisper-based Automatic Speech Recognition (ASR) model on Indian Supreme Court hearings.

---

## 📁 Project Structure

```
.
🔗 scraper.py                 # Downloads audio from YouTube and transcripts from PDF links
🔗 aligner.py                 # Aligns transcript with audio using forced alignment
🔗 process_dataset_upload_hf.py  # Prepares dataset in CommonVoice format and uploads to HF Hub
🔗 train.py                   # Fine-tunes the Whisper model on the aligned dataset
🔗 requirements.txt           # Required dependencies
🔗 final_data/                # Directory for processed training audio/text pairs
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd Adalat-AI-assignment
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🚀 Pipeline Steps

### 1. Scrape PDFs and Audio Files

Downloads court hearing audio from YouTube (as 16kHz mono `.wav`) and transcripts from linked PDFs.

```bash
python scraper.py
```

Use `scraper_mp.py` for scraping with multiple workers parallely. 

---

### 2. Align Transcripts with Audio

Performs silence removal using VAD, cleans the transcript, and generates aligned sentence-level timestamps. Also creates 5–30s training chunks.

```bash
python aligner.py
```

---

### 3. Process Dataset and Upload to HuggingFace

Formats dataset to [CommonVoice format](https://commonvoice.mozilla.org/en/datasets) and saves it locally.
To upload the dataset to Hugging Face Hub, add the `--push-to-hub` flag.

```bash
python process_dataset_upload_hf.py --push-to-hub
```

---

### 4. Train Whisper Model

```bash
python train.py \
  --model_name openai/whisper-medium \
  --language en \
  --sampling_rate 16000 \
  --num_proc 10 \
  --train_strategy epoch \
  --learning_rate 5e-5 \
  --warmup 50 \
  --train_batchsize 16 \
  --eval_batchsize 8 \
  --num_epochs 3 \
  --num_steps 1000 \
  --resume_from_ckpt None \
  --output_dir train_medium_v2_16_5e-5 \
  --train_datasets akshat311/supreme-court-stt \
  --train_dataset_splits train \
  --train_dataset_text_columns sentence \
  --eval_datasets akshat311/supreme-court-stt \
  --eval_dataset_splits test \
  --eval_dataset_text_columns sentence
```

---


## 🧠 Example Output Format (CommonVoice-style)

```json
{
  "path": "final_data/audio_4_003.wav",
  "audio": "final_data/audio_4_003.wav",
  "sentence": "This is a sample utterance from court hearing",
  "video_id": 4,
  "chunk_id": 3,
  "case_name": "XYZ vs State",
  "case_number": "SC/1234/2023",
  "hearing_date": "2023-01-01"
}
```

---

## 🧰 Model Used

* [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) — balanced for inference speed, accuracy, and multilingual Indian audio support.

---

