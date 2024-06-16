"""
File: text_extract.py
Author: Veronika Nevarilova (xnevar00)
Year: 2024
Description: This script extracts transcriptions from .xml.trans files for
       adaptation of automatic transcription model used for annotations.
"""

import os
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
                                                  directory passed via command line argument. It places trancsription \
                                                  of each segment in XML on a single line. \
                                                  Outputs are .txt files stored in subdirectory [dir]_extracted_[full | short] \
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
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_directory, args.dir)

    if not os.path.exists(data_directory):
        print(f"Directory '{args.dir}' does not exist.")
        exit()

    # creating output directory
    output_directory = os.path.join(current_directory, f"{args.dir}_extracted" + "_" + mode)
    os.makedirs(output_directory, exist_ok=True)

    # iterating through all transcription files
    for filename in os.listdir(data_directory):
        if filename.endswith(".xml.trans"):
            input_file_path = os.path.join(data_directory, filename)

            # getting name of output file - the same name as input file, but .txt
            output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0][:-4] + ".txt")

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                try:
                    tree = ET.parse(input_file_path)
                    root = tree.getroot()
                except:
                    print("Could not parse file " + input_file_path)

                # extracting text from all segments
                for segment in root.findall('.//segment'):
                    transcript = get_segment_text(segment, mode)

                    output_file.write(transcript + '\n')

            print(f"Text from file '{input_file_path}' has been extracted.")