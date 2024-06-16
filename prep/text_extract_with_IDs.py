"""
File: text_extract_with_IDs.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Decription: This script extracts transcriptions from .xml.trans files with
       adding the pseudoID of ATCo in the recording and saves it as a JSON.
"""

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from extract_fnc import set_mode, get_segment_text

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='This script extracts transcription from .xml.trans file stored in \
                                                  directory passed via command line argument. It creates a JSON file with \
                                                  transcriptions each on single line and pseudoID of ATCo present in the recording \
                                                  based on speaker label in XML file. \
                                                  Outputs are .json files stored in subdirectory [dir]_extracted_[full | short] \
                                                  in the same directory as the input directory.')
    parser.add_argument('dir', type=str, help='Subdirectory where all the input files are stored.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--short', action='store_true', help='The output is in shortened transcript format.')
    group.add_argument('-f', '--full', action='store_true', help='The output is in full transcript format.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    mode = set_mode(args)

    # getting data directory
    current_directory = os.path.dirname(__file__)
    data_directory = os.path.join(current_directory, args.dir)

    if not os.path.exists(data_directory):
        print(f"Directory '{args.dir}' does not exist.")
        exit()

    dir = args.dir.rstrip('/')
    print(dir)
    output_directory = os.path.join(current_directory, f"{dir}_extracted_jsons" + "_" + mode)
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(data_directory):
        if filename.endswith(".xml.trans"):
            input_file_path = os.path.join(data_directory, filename)

            # getting name of output file - the same name as input file, but .json
            output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0][:-4] + ".json")

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                try:
                    tree = ET.parse(input_file_path)
                    root = tree.getroot()
                except:
                    print("Could not parse file " + input_file_path)

                ATCoID = None
                language = None
                sentence = ""

                # extracting text from all segments
                segments = root.findall('.//segment')
                for segment in segments:
                    transcript = get_segment_text(segment, mode)

                    # checking for ATCo pseudoID
                    speaker_label = segment.find('speaker_label')

                    # if there is an ATCo ID present in segment, set his ID, else remain None
                    if isinstance(speaker_label.text, str):
                        match = re.match(r'ATCO tower ([0-9][0-9])$', speaker_label.text)
                        if match:
                            ATCoID = match.group(1)

                    sentence += transcript + '\n'

                # save extraction from current recording
                data = [{"sentence": sentence, "ATCo_ID": ATCoID}]
                json.dump(data, output_file, ensure_ascii=False, indent=4)

            print(f"Text and ATCoID from file '{input_file_path}' has been extracted.")