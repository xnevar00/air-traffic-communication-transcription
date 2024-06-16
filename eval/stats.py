"""
File: stats.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: Script for getting statistics of different sets from raw SpokenData data
"""

import glob
from xml.etree import ElementTree as ET
import re
import argparse
from typing import List

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="This script computes statistics of selected filtered set from raw SpokenData data.")
    parser.add_argument('directory', type=str, help='Directory with raw SpokenData data (containing .xml.trans files).')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-f', '--full', action='store_true', help='Work with full format.')
    group_mode.add_argument('-s', '--short', action='store_true', help='Work with shortened format.')

    group_filter = parser.add_mutually_exclusive_group(required=True)
    group_filter.add_argument('-u', '--unseen', action='store_true', help='Process only the unseen ATCOs segments.')
    group_filter.add_argument('-a', '--all', action='store_true', help='Process all files and segments in folder.')
    group_filter.add_argument('-o', '--others', action='store_true', help='Process only files with unseen ATCOs, but take only the segments without them.')
    group_filter.add_argument('-i', '--inverse', action='store_true', help='Process all files without unseen ATCOs.')

    args = parser.parse_args()
    return args

def get_full_transcript(text: str) -> str:
    """
    Extracts the full format from raw transcription text by removing text
    that precedes brackets + both brackets

    Returns:
        str: full format of transcription
    """
    return re.sub(r'(\S*)\s*\(([^)]*)\)', r'\2', text)

def get_shortened_transcript(text: str) -> str:
    """
    Extracts the shortened format from raw transcription text by removing both
    brackets + text between them

    Returns:
        str: short format of transcription
    """
    return re.sub(r'\([^)]*\)', '', text)

def remove_brackets_and_count_words(text: str) -> int:
    """
    Removes all [] brackets and stuff in them and count only the number of the
    rest of the words

    Returns:
        int: word count of text
    """
    cleaned_text = re.sub(r'\[.*?\]', '', text)     # removing [] brackets
    words = re.split(r'\s+', cleaned_text)
    words = [word for word in words if word]        # removing empty words

    return len(words)

def remove_specific_strings(strings):
    """
    Removes all redundant shortened versions of basic call signs (xx-xxx) already present
    in order to have more realistic view on number of call signs

    Returns:
        int: word count of text
    """
    patterns = {}
    strings = [item for item in strings if item is not None]

    for s in strings:
        if len(s) == 3: # shortened versions of normal planes usually have 3 characters
            key = (s[0], s[1:])
            patterns.setdefault(key, []).append(s)
        else:
            first, last_two = s[0], s[-2:] # if the shortened version is present in some full version, it is most likely the same plane
            if len(last_two) == 2:
                patterns.setdefault((first, last_two), []).append(s)

    result = []
    for s in strings:
        if len(s) == 3:
            key = (s[0], s[1:])
            if len(patterns.get(key, [])) > 1 and s in patterns[key]:
                continue
        result.append(s)
    return result

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

def print_table(data):
    print("+----------------------------------------+")
    print("|        WAV count:           |", end="")
    print(str(data["wav_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|     broadcast count:        |", end="")
    print(str(data["brdcst_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|       Czech count:          |", end="")
    print(str(data["czech_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|      English count:         |", end="")
    print(str(data["english_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|       Slovak count          |", end="")
    print(str(data["slovak_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|        words count          |", end="")
    print(str(data["words_count"]).center(10), end="|\n")
    print("+----------------------------------------+")
    print("|         seconds             |", end="")
    print(f"{data['seconds']:.2f}".center(10), end="|\n")
    print("+----------------------------------------+")
    print("|        speakers             |", end="")
    print(str(len(data["speakers"])).center(10), end="|\n")
    print("+----------------------------------------+")



if __name__ == "__main__":
    args = parse_arguments()

    data = {"wav_count": 0,
            "brdcst_count": 0,
            "czech_count": 0,
            "slovak_count": 0,
            "english_count": 0,
            "words_count": 0,
            "seconds": 0,
            "speakers": []}

    used_wav = False
    unseen_ATCOs = ["ATCO tower 01", "ATCO tower 06", "ATCO tower 07", "ATCO tower 08"]
    xml_files = glob.glob(f"{args.directory}/*.xml.trans")

    for file in xml_files:
        used_wav = False
        tree = ET.parse(file)
        root = tree.getroot()
        process_file = False
        segments_to_process = []
        transcripts = []

        # searching for files where there is ATCO from testset at least once among segments (broadcasts)
        if (args.others or args.unseen) and (check_for_unseen_ATCO(root, unseen_ATCOs) == False):
            continue

        # if we want to take only recordings without testset ATCO but in this recording he is present, skip the recording
        if (args.inverse) and (check_for_unseen_ATCO(root, unseen_ATCOs) == True):
            continue

        for segment in root.findall('segment'):
            speaker_label = segment.find('speaker_label').text
            start = float(segment.find('start').text) * 1000
            end = float(segment.find('end').text) * 1000
            sentence = segment.find('text').text

            if (args.short):
                sentence = get_shortened_transcript(sentence)
            else:
                sentence = get_full_transcript(sentence)

            language = determine_language(segment)

            ATCO_pattern = re.compile(r'ATCO tower \d{2}')

            if speaker_label and re.match(ATCO_pattern, speaker_label):
                ATC_ID = speaker_label[-2:]
            else:
                ATC_ID = None

            if ((args.unseen) and (speaker_label in unseen_ATCOs)) \
                or (args.all) or (args.inverse) \
                or (args.others and (speaker_label not in unseen_ATCOs)):  

                segments_to_process.append((start, end))

                # if the call sign was not present yet
                if (speaker_label not in data["speakers"]):
                    data["speakers"].append(speaker_label)

                data["words_count"] += remove_brackets_and_count_words(sentence)
                if (remove_brackets_and_count_words(sentence) == 0):
                    print("Empty transcription, skipping.")
                    continue
                
                used_wav = True
                data["brdcst_count"] +=1
                if (language == 'cs'):
                    data["czech_count"] +=1
                elif (language == 'sk'):
                    data["slovak_count"] +=1
                elif (language == 'en'):
                    data["english_count"] +=1

                data["seconds"] += (end - start)/1000
        if (used_wav):
            data["wav_count"] +=1

    speakers = remove_specific_strings(data["speakers"])
    print_table(data)