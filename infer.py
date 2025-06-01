# import torch
# from transformers import pipeline
# from datasets import load_dataset

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# pipe = pipeline(
#   "automatic-speech-recognition",
#   model="openai/whisper-medium",
# #   chunk_length_s=30,
#   device=device,
# )
# pipe.model.config.forced_decoder_ids = (
#         pipe.tokenizer.get_decoder_prompt_ids(
#             language="en", task="transcribe"
#         )
#     )
# pipe.model.generation_config.forced_decoder_ids = (
#     pipe.tokenizer.get_decoder_prompt_ids(
#         language="en", task="transcribe"
#     )
# )
# audio = "/root/personal/Adalat-AI-assignment/final_data/audio_16_490.wav"
# # we can also return timestamps for the predictions
# prediction = pipe(audio)
# print(prediction)




from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
dataset = load_dataset("akshat311/supreme-court-stt", split="test")
sample = dataset[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features, language = "en")
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

print(transcription)