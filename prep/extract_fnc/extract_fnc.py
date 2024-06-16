import re
import argparse
import xml.etree.ElementTree as ET

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

def set_mode(args: argparse.Namespace) -> str:
    """
    Sets the mode of transcription - full or shortened

    Returns:
        str: short format of transcription
    """
    if args.full:
        mode = 'f'
    elif args.short:
        mode = 's'
    
    return mode

def get_segment_text(segment: ET.Element, mode: str) -> str:
    text_element = segment.find('text')
    if text_element is not None and text_element.text is not None:
        text = text_element.text.strip()

        if (mode == "s"):
            cleaned_text = get_shortened_transcript(text)
        else:
            cleaned_text = get_full_transcript(text)

    return cleaned_text