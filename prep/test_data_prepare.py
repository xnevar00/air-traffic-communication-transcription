"""
File: test_data_prepare.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: This script takes recordings, filters them based on selected mode,
             cuts them to single broadcasts and sets corresponding language.
"""

import glob
from xml.etree import ElementTree as ET
from pydub import AudioSegment
import json
import os
import re
import argparse
from extract_fnc import get_shortened_transcript, get_full_transcript
from typing import List, Tuple

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="This script takes recordings, filters them based on selected mode, \
                                                  cuts them to single broadcasts and sets corresponding language.")
    parser.add_argument('directory', type=str, help='Directory with raw SpokenData data (containing .wav and .xml.trans files).')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-f', '--full', action='store_true', help='Transcripts are in full format.')
    group_mode.add_argument('-s', '--short', action='store_true', help='Transcripts are in shortened format.')

    group_filter = parser.add_mutually_exclusive_group(required=True)
    group_filter.add_argument('-u', '--unseen', action='store_true', help='Process only the unseen ATCOs broadcasts and add a language to it.')
    group_filter.add_argument('-a', '--all', action='store_true', help='Process all broadcasts and add a language to it.')
    group_filter.add_argument('-o', '--others', action='store_true', help='Process only files with unseen ATCOs, but take only the broadcasts without them.')

    args = parser.parse_args()
    return args

def check_for_unseen_ATCO(root: ET.ElementTree, unseen_ATCOs: List[str]) -> bool:
    """
    Checks wether there is a ATCO from test set in anz segment of transcription

    Parameters:
        root (ET.ElementTree): root for whole XML
        unseen_ATCOs (List[str]): list with IDs of unseen ATCOs
    """
    for segment in root.findall('segment'):
        speaker_label = segment.find('speaker_label').text
        if speaker_label in unseen_ATCOs:
            return True

    return False

def determine_language(segment: ET.ElementTree) -> str:
    """
    Determines the language of broadcast segment based on non_english and in-text tag

    Returns:
        str: abbreviation of language
    """
    non_english_element = segment.find('.//tags/non_english')
    if non_english_element is not None and isinstance(non_english_element.text, str):
        if (non_english_element.text == "0"):
            language = "en"
        else:
            text_element = segment.find('text')
            words = text_element.text.strip().split()
            if words and words[0] == "[Slovak]":
                language = "sk"
            else:
                language = "cs"
    return language

def process_recording(file: str, args: argparse.Namespace, segments_to_process: List[Tuple[float, float]], transcripts: List[dict], raw_data_path):
    """
    Cuts WAV audio file based on start and end timestamps in segments_to_process
    and saves info about it with transcription as JSON

    Parameters:
        file (str): path to .xml.trans file containing transcript and other info
        args (argparse.Namespace): CL arguments to determine mode
        segments_to_process (List[Tuple[float, float]]): list with tuples of broadcast timestamps
        transcripts (List[dict]): list of dictionnaries with info about transcription, speaker and language
    """
    wav_file_name = file.replace(".xml.trans", ".wav")
    wav_file_name_with_extension = os.path.basename(wav_file_name)
    output_wav_file_name, _ = os.path.splitext(wav_file_name_with_extension)

    if args.full:
        mode_folder = "full"
    else:
        mode_folder = "short"

    audio = AudioSegment.from_wav(f"{wav_file_name}")

    for i, ((start, end), data) in enumerate(zip(segments_to_process, transcripts), start=1):
        segment_audio = audio[start:end]
        result_file_path = raw_data_path + f"/audio/{output_wav_file_name}_segment{i}.wav"
        segment_audio.export(result_file_path, format="wav")

        json_file_path = raw_data_path + f"/transcripts_{mode_folder}/{output_wav_file_name.replace('.wav', '')}_segment{i}.json"
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump([data], json_file, ensure_ascii=False, indent=4)

def initialize_output_dirs(args: argparse.Namespace):
    """
    Parses the command line arguments.

    Parameters:
        args (argparse.Namespace): CL arguments containing mode of transcript
    """
    if args.full:
        mode_folder = "full"
    else:
        mode_folder = "short"

    full_directory_path = os.path.abspath(args.directory)
    root_directory = os.path.dirname(full_directory_path)
    raw_data_path = os.path.join(root_directory, 'test_datasets_data')
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    audio_folder_path = raw_data_path + f"/audio"
    result_folder_path = raw_data_path + f"/transcripts_{mode_folder}"

    os.makedirs(audio_folder_path, exist_ok=True)
    os.makedirs(result_folder_path, exist_ok=True)

    return raw_data_path

if __name__ == '__main__':
    args = parse_arguments()

    unseen_ATCOs = ["ATCO tower 01", "ATCO tower 06", "ATCO tower 07", "ATCO tower 08"]
    xml_files = glob.glob(f"{args.directory}/*.xml.trans")
    raw_data_path = initialize_output_dirs(args)

    for file in xml_files:
        tree = ET.parse(file)
        root = tree.getroot()

        segments_to_process = []
        transcripts = []

        # searching for files where there is ATCO from testset at least once among segments (broadcasts)
        if (args.others or args.unseen) and (check_for_unseen_ATCO(root, unseen_ATCOs) == False):
            continue

        for segment in root.findall('segment'):
            speaker_label = segment.find('speaker_label').text
            sentence = segment.find('text').text
            start = float(segment.find('start').text) * 1000 # for cutting purpose
            end = float(segment.find('end').text) * 1000

            if (args.short):
                sentence = get_shortened_transcript(sentence)
            else:
                sentence = get_full_transcript(sentence)

            language = determine_language(segment)

            ATCO_pattern = re.compile(r'ATCO tower \d{2}')

            # if the speaker in a segment is an ATCO from testset, get his ID
            if speaker_label and re.match(ATCO_pattern, speaker_label):
                ATC_ID = speaker_label[-2:]
            else:
                ATC_ID = None

            # determining based on selected mode
            if ((args.unseen) and (speaker_label in unseen_ATCOs)) \
                or (args.all) \
                or (args.others and (speaker_label not in unseen_ATCOs)):
                    
                    # add this broadcast for processing
                    segments_to_process.append((start, end))
                    transcripts.append({
                        "sentence": sentence,
                        "ATCo_ID": ATC_ID,
                        "language": language
                    })

        process_recording(file, args, segments_to_process, transcripts, raw_data_path)
