"""
File: call_sign_error_rate.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: Script for evaluating error rate on call signs
"""

import argparse

wer_data = {'sub' : 0, 'ins' : 0, 'dels' : 0, 'cor' : 0}

def parse_arguments():
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--file', required=True, help='File with sentences and transcriptions')

    args = parser.parse_args()
    return args


def getAnswer():
    print("Enter S I D C:", end="")
    user_input = input()
    print("\n")

    segments = user_input.split('/')

    for segment in segments:
        numbers = segment.strip().split()
        # count all the watched metrics
        if (len(numbers) == 1):
            wer_data['cor'] += int(numbers[0])
        elif len(numbers) == 4:
            wer_data['sub'] += int(numbers[0])
            wer_data['ins'] += int(numbers[1])
            wer_data['dels'] += int(numbers[2])
            wer_data['cor'] += int(numbers[3])
        else:
            return

def computeWER():
    #compute WER from all data
    value = (wer_data['sub'] + wer_data['dels'] + wer_data['ins'])/(wer_data['sub'] + wer_data['dels'] + wer_data['cor'])
    print("Total WER: ", value)


arguments = parse_arguments()
file_path = arguments.file
with open(file_path, 'r', encoding='utf-8') as file:
    current_sentence = None
    for line in file:
        if line.startswith('Sentence'):
            current_sentence = line.strip()
        elif line.startswith('Transcription') and current_sentence:
            print(current_sentence)
            print(line.strip())
            getAnswer()
            current_sentence = None  # reset for next pair
    computeWER()