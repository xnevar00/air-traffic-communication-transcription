"""
File: wer_ev_SpokenData.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: Script for evaluating WER on different testsets
"""

import argparse
from evaluate import load
from xml.etree import ElementTree as ET
import glob
import os
import re

def parse_arguments():
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--raw_SpokenData_dir', required=True, help='Directory containing raw data from SpokenData.')

    args = parser.parse_args()
    return args

def get_full_transcript(text: str):
    """
    Removes text that precedes the brackets + both brackets
    """
    text = re.sub(r'\[.*?\]', '', text)
    return re.sub(r'(\S*)\s*\(([^)]*)\)', r'\2', text)

def checkForLanguage(sentence: str) -> str:
    """
    Determines the language - Czech or Slovak

    Returns:
        str: language specification
    """
    all_slovak = True
    none_slovak = True

    for segment in sntc_root.findall('.//segment'):
        text = segment.find('text').text.strip()
        if text.startswith("[slovak]"):
            none_slovak = False
        else:
            all_slovak = False

    if all_slovak:
        return "sk"
    elif none_slovak:
        return "cs"
    
def printTable(wer_data, name):
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



arguments = parse_arguments()
dir = arguments.raw_SpokenData_dir

xml_files = glob.glob(os.path.join(dir, '*.xml'))

predictions_cs = []
predictions_en = []
predictions_sk = []
predictions_all = []

references_cs = []
references_en = []
references_sk = []
references_all = []


file_number = 0
for xml_file in xml_files:
    file_number += 1

    tree = ET.parse(xml_file)
    root = tree.getroot()

    preds = [segment.find('text').text.strip() for segment in root.findall('.//segment')]
    # getting transcriptions predicted by SpokenData and connecting them into one string
    preds_whole = '\n'.join(preds)

    sntc_file = xml_file + ".trans"
    if os.path.exists(sntc_file):
        sntc_tree = ET.parse(sntc_file)
        sntc_root = sntc_tree.getroot()
        non_english_values = set([segment.find('.//non_english').text for segment in sntc_root.findall('.//segment')])
    
        # taking only recordings with the same language through all the recording
        if len(non_english_values) == 1:
            non_english = non_english_values.pop()

            sntc = [segment.find('text').text.strip() for segment in sntc_root.findall('.//segment')]
            # getting manual sentence transcriptions and connecting them into one string
            sntc_whole = '\n'.join(sntc)
            # extracting only full transcript of the sentence because SpokenData transcripts only the full version
            sntc_whole = get_full_transcript(sntc_whole)
            if (sntc_whole == "" or sntc_whole == None or preds_whole == "" or preds_whole == None):
                continue

            if non_english == "1":
                language = checkForLanguage(sntc)
                if (language == "cs"):
                    references_cs.append(sntc_whole)
                    predictions_cs.append(preds_whole)
                elif (language == "sk"):
                    references_sk.append(sntc_whole)
                    predictions_sk.append(preds_whole)
                else:
                    continue
            else:
                references_en.append(sntc_whole)
                predictions_en.append(preds_whole)

            references_all.append(sntc_whole)
            predictions_all.append(preds_whole)

            print(f"Sentence {file_number}: ", sntc_whole, "\n")
            print(f"Transcription {file_number}: ", preds_whole, "\n")


wer = load("wer")

result_score = {
    "count_overall": len(predictions_all), 
    "count_cs": len(predictions_cs), 
    "count_en": len(predictions_en), 
    "count_sk": len(predictions_sk),
    "wer_overall": 0 if len(predictions_all) == 0 else wer.compute(predictions=predictions_all, references=references_all),
    "wer_cs": 0 if len(predictions_cs) == 0 else wer.compute(predictions=predictions_cs, references=references_cs),
    "wer_en": 0 if len(predictions_en) == 0 else wer.compute(predictions=predictions_en, references=references_en),
    "wer_sk": 0 if len(predictions_sk) == 0 else wer.compute(predictions=predictions_sk, references=references_sk)
}

printTable(result_score, "SpokenData")

