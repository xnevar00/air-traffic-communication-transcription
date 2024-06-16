"""
File: dataset_prepare.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: This script makes dataset from given audio and transcription.
       Based on selected mode, it creates a trainset or testset (with added language)
"""

import os
import json
import numpy as np
import soundfile as sf
import re
import argparse
from typing import Tuple
import datetime

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="This script makes dataset from given audio and transcription. \
                                     It runs in two modes: mode for train dataset (it skips ATCos selected for testset) \
                                     and mode for test dataset with added language (making a dataset from all given data).  \
                                     Output is in 'dataset' folder which is in the same folder as input json folder ")
    parser.add_argument('audio_directory', type=str, help='Directory with .wav files.')
    parser.add_argument('json_directory', type=str, help='Directory with .json files containing output from text_extract_with_IDs.py.\
                                                          Script pairs the audio and json file based on same name.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='Set mode for train dataset - skip ATCos from testset.')
    group.add_argument('-v', '--test', action='store_true', help='Set mode for test dataset - make dataset from all data.')

    args = parser.parse_args()
    return args

def get_json_filename(audio_filename: str) -> str:
    """
    Parses the audio filename and makes a JSON filename from it

    Parameters:
        audio_filename (str): name of the WAV file

    Returns:
        str: name of JSON file containing transcription
    """
    # transcript has the same name as the .wav file
    file_name_without_extension = os.path.splitext(audio_filename)[0]
    json_filename = os.path.join(args.json_directory, file_name_without_extension + '.json')

    return json_filename

def process_text(json_filename: str, args: argparse.Namespace) -> Tuple[str, str, str]:
    """
    Parses the JSON file

    Parameters:
        json_filename (str):        name of the JSON file
        args (argparse.Namespace):  CL arguments to get info about type of the dataset (train/test)

    Returns:
        Tuple[str, str, str]: extracted transcription, atcID and language (if testset)
    """
    # getting the reference transcript and ID of ATCo
    with open(json_filename, 'r', encoding='utf-8') as text_file:
        data = json.load(text_file)
        atcID = data[0]['ATCo_ID']
        sentence = data[0]['sentence']
        language = None

        # removing tags from SpokenData
        sentence = re.sub(r'\[.*?\]', '', sentence)

        # skipping possible empty transcriptions
        if (sentence == "" or sentence.isspace()):
            return None, None, None
    
    # skipping test ATCos in trainset mode
    if (args.train) and (atcID in test_ATCOs):
        return None, None, None
    elif args.test:
        # testsets have additional language info
        language = data[0].get('language')
        if not language:
            print("JSON " + json_filename + " does not have 'language' key which is mandatory for testset.")
            exit()
        
    return sentence, atcID, language

def process_audio(audio_path: str) -> dict:
    """
    Loads audio and extracts information about it

    Parameters:
        audio_path (str): path to WAV file

    Returns:
        dict: info about audio
    """
    # getting the audio to list and the sampling frequency
    orig, Fs = sf.read(audio_path)
    audio_data = {
        'path': audio_path,
        'array': orig.tolist(),
        'sampling_rate': Fs
    }

    return audio_data

if __name__ == '__main__':
    args = parse_arguments()

    # pseudoIDs of ATCos selected for testset
    test_ATCOs = ["01", "06", "07", "08"]

    output_data = []

    # going through all .wav files in the audio_directory
    for audio_filename in os.listdir(args.audio_directory):
        if audio_filename.endswith('.wav'):

            audio_filepath = os.path.join(args.audio_directory, audio_filename)

            json_filename = get_json_filename(audio_filename)

            # if there is .txt transcript for the particular .wav file
            if not os.path.exists(json_filename):
                print("Could not find a matching transcript for " + audio_filename + ", skipping.")
                continue

            sentence, atcID, language = process_text(json_filename, args)
            if (sentence is None):
                continue

            audio_data = process_audio(audio_filepath)
            
            # creating json object, one for every pair audio + its transcript
            json_data = {
                'audio': audio_data,
                'sentence': sentence,
            }

            if args.test:
                json_data['language'] = language
            
            # 09 is unknown ATCO, skipping
            if (atcID != "09"):
                if (args.train) and (atcID not in test_ATCOs):
                    output_data.append(json_data)
                elif (args.test):
                    output_data.append(json_data)
                else:
                    continue
            else:
                continue
      
    # save the datasets
    time = datetime.datetime.now().isoformat().replace(":", "").split('.')[0]
    if (args.train):
        dataset_name = f'train_dataset_{time}.json'
        dir_name = 'train_datasets'
    else:
        dataset_name = f'test_dataset_{time}.json'
        dir_name = 'test_datasets'

    # saving the dataset to dir 'datasets' in same root dir as the json dir
    full_json_directory_path = os.path.abspath(args.json_directory)
    root_directory = os.path.dirname(full_json_directory_path)

    datasets_path = os.path.join(root_directory, dir_name)
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    dataset_file_path = os.path.join(datasets_path, dataset_name)

    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("Datasets were successfully created.")
    print("Number of records: " + str(len(output_data)))