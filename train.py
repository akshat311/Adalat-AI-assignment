import numpy as np
import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets, load_from_disk
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
# torch.set_num_threads(1)
import wandb
wandb.init(project="whisper")
#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)


parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=2, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--language', 
    type=str, 
    required=False, 
    default='en',
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=48, 
    help='Batch size during the training phase.'
)

parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=5, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--train_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of training dataset splits. Eg. 'train' for the train split of Common Voice",
)
parser.add_argument(
    '--train_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each training dataset. Eg. 'sentence' for Common Voice",
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for evaluation.'
)
parser.add_argument(
    '--eval_dataset_splits', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="List of evaluation dataset splits. Eg. 'test' for the test split of Common Voice",
)
parser.add_argument(
    '--eval_dataset_text_columns', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help="Text column name of each evaluation dataset. Eg. 'transcription' for Google Fleurs",
)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

if len(args.train_datasets) == 0:
    raise ValueError('No train dataset has been passed')
if len(args.eval_datasets) == 0:
    raise ValueError('No evaluation dataset has been passed')

if len(args.train_datasets) != len(args.train_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_splits. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_splits)} for train_dataset_splits.")
if len(args.eval_datasets) != len(args.eval_dataset_splits):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_splits. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_splits)} for eval_dataset_splits.")

if len(args.train_datasets) != len(args.train_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_text_columns. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_text_columns)} for train_dataset_text_columns.")
if len(args.eval_datasets) != len(args.eval_dataset_text_columns):
    raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_text_columns. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_text_columns)} for eval_dataset_text_columns.")

print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
print('ARGUMENTS OF INTEREST:')
print(vars(args))
print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


#############################       MODEL LOADING       #####################################
from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

# if freeze_feature_encoder:
#     model.freeze_feature_encoder()

# if freeze_encoder:
#     model.freeze_encoder()
#     model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
# model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False


############################        DATASET LOADING AND PREP        ##########################

def load_all_datasets(split):    
    combined_dataset = []
    if split == 'train':
        for i, ds in enumerate(args.train_datasets):
            lang = args.language
            dataset = load_dataset(ds, split=args.train_dataset_splits[i])
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.train_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.train_dataset_text_columns[i], "sentence")
            # dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            language_column = [lang] * len(dataset)
            dataset = dataset.add_column("language", language_column)
            # dataset = dataset.select(range(min(10000, len(dataset)))) if args.train_dataset_splits[i] == 'train' else dataset.select(range(1000))
            combined_dataset.append(dataset)
    elif split == 'eval':
        for i, ds in enumerate(args.eval_datasets):
            lang = args.language
            dataset = load_dataset(ds, split=args.eval_dataset_splits[i]).select(range(50))
            dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
            if args.eval_dataset_text_columns[i] != "sentence":
                dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
            # dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            language_column = [lang] * len(dataset)
            dataset = dataset.add_column("language", language_column)
            # dataset = dataset.select(range(min(1000, len(dataset))))
            combined_dataset.append(dataset)
    
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)
    print(len(ds_to_return))
    return ds_to_return


max_label_length = model.config.max_length
print(f"Max label length: {max_label_length}")
min_input_length = 0.0
max_input_length = 30.0



print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_all_datasets('train')
raw_dataset["eval"] = load_all_datasets('eval')

#shuffle the dataset
raw_dataset["train"] = raw_dataset["train"].shuffle(seed=22)


###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    do_lower_case: bool = False
    do_remove_punctuation: bool = False
    normalizer: BasicTextNormalizer = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input features from audio
        input_features = [
            self.processor.feature_extractor(
                feature["audio"]["array"],
                sampling_rate=feature["audio"]["sampling_rate"]
            ).input_features[0] for feature in features
        ]
        input_features = np.array(input_features, dtype=np.float32)  # Ensure consistent data type
        batch = {"input_features": torch.tensor(input_features)}


        # Preprocess transcriptions
        label_features = []
        for feature in features:
            transcription = feature["sentence"]
            if self.do_lower_case:
                transcription = transcription.lower()
            if self.do_remove_punctuation and self.normalizer:
                transcription = self.normalizer(transcription).strip()
            self.processor.tokenizer.set_prefix_tokens(language=feature["language"], task="transcribe")
            input_ids = self.processor.tokenizer(transcription).input_ids
            if len(input_ids) > max_label_length:
                print(self.processor.tokenizer.decode(input_ids))
                input_ids = input_ids[:max_label_length]
                
            label_features.append({"input_ids": input_ids})

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove initial BOS token if necessary
        if (labels[:, 0] == 50258).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, do_lower_case=do_lower_case, do_remove_punctuation=do_remove_punctuation, normalizer=normalizer)
print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    # label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_ids[label_ids == -100] = 50256

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    wer_norm = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    cer_norm = 100 * cer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
    return {"wer": wer, "wer_norm": wer_norm, "cer": cer, "cer_norm": cer_norm}


###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        # evaluation_strategy="no",
        eval_steps=1800,
        save_strategy="steps",
        save_steps=200,
        max_steps=args.num_steps,
        save_total_limit=30,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        # optim="adamw_bnb_8bit",
        # resume_from_checkpoint=args.resume_from_ckpt,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()
# trainer.train(resume_from_checkpoint=args.resume_from_ckpt)
print('DONE TRAINING')


