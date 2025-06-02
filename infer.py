from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None

dataset = load_dataset("akshat311/supreme-court-stt", split="test")
sample = dataset[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

predicted_ids = model.generate(input_features, language = "en")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

print(transcription)