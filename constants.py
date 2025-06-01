from dataclasses import dataclass

@dataclass
class AlignerConfig:
    language: str = "eng"  # ISO-639-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    audio_dir: str = "audio_files"
    transcript_dir: str = "transcripts"
    model_name: str = "facebook/wav2vec2-large-960h"
    split_size: str = "sentence"
    