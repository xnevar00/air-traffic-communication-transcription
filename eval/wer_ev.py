"""
File: wer_ev.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: Script for evaluating WER on different testsets
"""

# because while importing transformers library CUDA is initialized and therefore it cannot be claimed by safe-gpu,
# the claim must be in here before imports
from safe_gpu import safe_gpu
safe_gpu.claim_gpus()
print("GPUs claimed.")

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import argparse
from evaluate import load

def parse_arguments():
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--processor', required=True, help='Directory with Whisper processor')
    parser.add_argument('--model', required=True, help='Directory with Whisper model to work with.')
    parser.add_argument('--unseen_ATC', required=False, help='Directory with unseen ATCOs dataset.')
    parser.add_argument('--unseen_airport', required=False, help='Directory with unseen airport dataset.')
    parser.add_argument('--unseen_LKKU', required=False, help='Directory with unseen LKKU data dataset.')

    args = parser.parse_args()
    return args

def computeWER(processor, model, wer, ds):
    """
    Computes WER for all languages in different testsets

    Parameters:
        processor: loaded Whisper processor
        model: loaded Whisper model
        wer: loaded wer
        ds: dataset on which to compute WER
    
    Returns:
        dictionnary with all partial WER values
    """

    predictions_all = []
    predictions_en = []
    predictions_cs = []
    predictions_sk = []

    references_all = []
    references_en = []
    references_cs = []
    references_sk = []


    for i in range(len(ds)):

        sample = ds[i]["audio"]
        reference = ds[i]["sentence"]

        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.to('cuda') 

        predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


        if (ds[i]["language"] == "en"):
            predictions_en.append(transcription)
            references_en.append(reference)
        elif (ds[i]["language"] == "cs"):
            predictions_cs.append(transcription)
            references_cs.append(reference)
        elif (ds[i]["language"] == "sk"):
            predictions_sk.append(transcription)
            references_sk.append(reference)

        predictions_all.append(transcription)
        references_all.append(reference)
        
        print(f"Sentence {i+1}: ", reference)
        print("Transcription: ", transcription)

    
    print(len(ds))
        
    result_score = {
        "count_overall": len(ds), 
        "count_cs": len(predictions_cs), 
        "count_en": len(predictions_en), 
        "count_sk": len(predictions_sk),
        "wer_overall": 0 if len(predictions_all) == 0 else wer.compute(predictions=predictions_all, references=references_all),
        "wer_cs": 0 if len(predictions_cs) == 0 else wer.compute(predictions=predictions_cs, references=references_cs),
        "wer_en": 0 if len(predictions_en) == 0 else wer.compute(predictions=predictions_en, references=references_en),
        "wer_sk": 0 if len(predictions_sk) == 0 else wer.compute(predictions=predictions_sk, references=references_sk)
    }

    return result_score

def printTable(wer_data: dict, name: str):
    """
    Prints all partial wer values arranged into table

    Parameters:
        wer_data: dict for specific testset
        name: name to be printed in the head of table
    """

    print("+-----------------------------------------------------------------------------------------+")
    print("|                                     Word Error Rate                                     |")
    print("|", end="")
    print(name.center(89), end="|\n")
    print("+-----------------+-----------------+-----------------------------------------------------+")
    print("|                 |                 |                       Details                       |")
    print("+   Total count   |    Total WER    +-----------------+-----------------+-----------------+")
    print("|                 |                 |     Language    |      Count      |       WER       |")
    print("+-----------------+-----------------+-----------------+-----------------+-----------------+")
    print("|                 |                 |      Czech      |", end="")
    print(f"{wer_data['count_cs']}".center(17), end="|")
    print(f"{wer_data['wer_cs']:.5f}".center(17), end="|\n")
    print("|                 |                 +-----------------+-----------------+-----------------+")
    print("|", end="")
    print(f"{wer_data['count_overall']}".center(17), end="|")
    print(f"{wer_data['wer_overall']:.5f}".center(17), end="|")
    print("     English     |", end="")
    print(f"{wer_data['count_en']}".center(17), end="|")
    print(f"{wer_data['wer_en']:.5f}".center(17), end="|\n")
    print("|                 |                 +-----------------+-----------------+-----------------+")
    print("|                 |                 |      Slovak     |", end="")
    print(f"{wer_data['count_sk']}".center(17), end="|")
    print(f"{wer_data['wer_sk']:.5f}".center(17), end="|\n")
    print("+-----------------+-----------------+-----------------+-----------------+-----------------+")


if __name__ == "__main__":

    arguments = parse_arguments()
    wer = load("wer")

    # load model and processor
    processor = WhisperProcessor.from_pretrained(arguments.processor)
    model = WhisperForConditionalGeneration.from_pretrained(arguments.model).to('cuda')
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe", language="czech")

    
    wer_score = {"unseen_ATC": None, "unseen_airport": None, "unseen_LKKU": None}

    if (arguments.unseen_ATC):
        ds = load_dataset(arguments.unseen_ATC, split="test")
        wer_score["unseen_ATC"] = computeWER(processor, model, wer, ds)
        printTable(wer_score["unseen_ATC"], "Unseen ATCos")

    if (arguments.unseen_airport):
        ds = load_dataset(arguments.unseen_airport, split="train")
        wer_score["unseen_airport"] = computeWER(processor, model, wer, ds)
        printTable(wer_score["unseen_airport"], "Unseen airport")

    if (arguments.unseen_LKKU):
        ds = load_dataset(arguments.unseen_LKKU, split="test")
        wer_score["unseen_LKKU"] = computeWER(processor, model, wer, ds)
        printTable(wer_score["unseen_LKKU"], "Unseen LKKU")
        

