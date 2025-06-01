import os
import argparse
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
normalizer = BasicTextNormalizer()

def normalize_text(batch):
    batch["norm_text"] = normalizer(batch["sentence"])
    return batch

def main(args):
    # Load model and processor
    processor = WhisperProcessor.from_pretrained("/".join(args.hf_model.split("/")[:-1]))
    model = WhisperForConditionalGeneration.from_pretrained(args.hf_model)
    model.to(f"cuda:{args.device}" if args.device >= 0 else "cpu")

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split).select(range(50))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalize_text)

    predictions = []
    references = []
    norm_predictions = []
    norm_references = []

    print("🧠 Running inference...")
    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(model.device)

        pred_ids = model.generate(input_features, language = "en")
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        predictions.append(pred_text)
        references.append(sample["sentence"])
        norm_predictions.append(normalizer(pred_text))
        norm_references.append(sample["norm_text"])

    # Compute metrics
    wer = round(100 * wer_metric.compute(predictions=predictions, references=references), 2)
    cer = round(100 * cer_metric.compute(predictions=predictions, references=references), 2)
    norm_wer = round(100 * wer_metric.compute(predictions=norm_predictions, references=norm_references), 2)
    norm_cer = round(100 * cer_metric.compute(predictions=norm_predictions, references=norm_references), 2)

    print("\n🎯 Results:")
    print(f"WER          : {wer}")
    print(f"CER          : {cer}")
    print(f"NORM WER     : {norm_wer}")
    print(f"NORM CER     : {norm_cer}")

    os.makedirs(args.output_dir, exist_ok=True)
    fname = args.dataset.replace('/', '_') + '_' + args.split + '_' + args.hf_model.replace('/', '_')
    with open(os.path.join(args.output_dir, fname), "w") as f:
        f.write(f"WER: {wer}\nCER: {cer}\nNORMALIZED WER: {norm_wer}\nNORMALIZED CER: {norm_cer}\n\n")
        for ref, hyp in zip(references, predictions):
            f.write(f"REF: {ref}\nHYP: {hyp}\n" + "-"*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_model", type=str, default="/root/personal/Adalat-AI-assignment/train_medium_v2_16_1e-6/checkpoint-596")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--dataset", type=str, default="akshat311/supreme-court-stt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)  # Not used, kept for compat
    parser.add_argument("--output_dir", type=str, default="predictions_dir")

    args = parser.parse_args()
    main(args)
