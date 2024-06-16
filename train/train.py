"""
File: train.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: A training script for Whisper. Main structure taken from https://huggingface.co/blog/fine-tune-whisper, modified.
"""

# because while importing transformers library CUDA is initialized and therefore it cannot be claimed by safe-gpu,
# the claim must be in here before imports
from safe_gpu import safe_gpu
safe_gpu.claim_gpus()
print("GPUs claimed.")

from datasets import DatasetDict, Audio, Dataset, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from functools import partial
import os
import json
import argparse
import logging

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def parse_arguments():
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--testset', required=True, help='Directory with test datasets (json files).')
    parser.add_argument('--trainset', required=True, help='Directory with train datasets (json files).')
    parser.add_argument('--feature_etc_files', required=True, help='Directory containing feature_extractor, tokenizer etc.')
    parser.add_argument('--model', required=True, help='Directory containing the model for learning.')
    parser.add_argument('--logging_dir', required=True, help='Directory for logging.')
    parser.add_argument('--training_output_dir', required=True, help='Directory where all checkpoints and other data will be stored.')
    parser.add_argument('--new_model_path', required=True, help='Directory where the trained model will be saved.')
    parser.add_argument('--lr', required=True, help='')
    parser.add_argument('--warmup', required=True, help='')
    parser.add_argument('--batch_size', required=True, help='')
    parser.add_argument('--num_epochs', required=True, help='')
    parser.add_argument('--freeze_encoder', required=False, help='')
    parser.add_argument('--dropout', required=False, help='')

    args = parser.parse_args()
    return args

def load_datasets_from_folder(folder_path: str, type: str) -> list:
    """
    Loads all JSON datasets from folder_path and concatenates them into one big dataset

    Parameters:
        folder_path (str): folder where all the datasets to be concatenated are stored
        type (str): 'train' or 'test'. If it is test, it also removes a 'language' key from it,
                    since it is there for WER evaluation of other script

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    datasets = []

    # making one big dataset from all jsons in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                data = json.load(file)

                # remove language values from testset
                if type == "test":
                    for item in data:
                        item.pop('language', None)

                dataset = Dataset.from_list(data)

            datasets.append(dataset)

    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Prepares a dataset by processing audio data and transcriptions.

    Parameters:
        batch: A dictionary containing the audio data and sentence.
        feature_extractor:  WhisperFeatureExtractor
        tokenizer: WhisperTokenizer

    Returns:
        dict: The modified batch dictionary that includes the processed audio features
              and tokenized sentence as label ids under the key "labels".
    """
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(raw_speech=audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def compute_metrics(pred):
    """
    Computes word error rate for model predictions against their reference.
    This function is called at every checkpoint of training.

    Parameters:
        pred: structure containing predictions and reference IDs

    Returns:
        dict: computed metrics
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":

    print("Parsing arguments..")
    arguments = parse_arguments()

    logging.basicConfig(level="INFO", force=True)

    print("Loading train dataset...")
    train_dataset = load_datasets_from_folder(arguments.trainset, "train")

    print("Loading test dataset...")
    test_dataset = load_datasets_from_folder(arguments.testset, "test")

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # print the structure of dataset to see how many data is in test and train set
    print(dataset)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(arguments.feature_etc_files)

    # important to set everything to czech, othervise it was not working
    tokenizer = WhisperTokenizer.from_pretrained(arguments.feature_etc_files, task="transcribe", language="czech")

    processor = WhisperProcessor.from_pretrained(arguments.feature_etc_files, task="transcribe", language="czech")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2, fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer})

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(arguments.model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()
    model.generate = partial(model.generate, task="transcribe", language="czech")
    model.config.suppress_tokens = []

    if (arguments.freeze_encoder):
        model.freeze_encoder()
    if (arguments.dropout):
        model.config.dropout = float(arguments.dropout)

    training_args = Seq2SeqTrainingArguments(
        output_dir=arguments.training_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
        learning_rate=float(arguments.lr),
        warmup_ratio=float(arguments.warmup),
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        per_device_eval_batch_size=int(arguments.batch_size),
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        num_train_epochs=int(arguments.num_epochs),
        logging_steps=30,
        logging_dir=arguments.logging_dir
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training..")
    trainer.train()
    trainer.save_model(arguments.new_model_path)
