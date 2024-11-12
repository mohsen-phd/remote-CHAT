import re
from typing import Optional
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk


# Function to expand contractions
def expand_contractions(text: str) -> str:
    """Expand contractions in the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with expanded contractions.
    """
    contractions_dict = {
        "'s": " is",  # father's -> father
        "'re": " are",  # you're -> you are
        "'ve": " have",  # I've -> I have
        "'ll": " will",  # you'll -> you will
        "'d": " would",  # he'd -> he would
        "n't": " not",  # can't -> cannot
        "'m": " am",  # I'm -> I am
        "'t": " not",  # doesn't -> does not (extra handling required to avoid overlap with "n't")
    }

    for contraction, replacement in contractions_dict.items():
        text = re.sub(rf"{contraction}\b", replacement, text)
    return text


def remove_contractions(text: str) -> str:

    contractions_dict = {
        "'s": "",  # father's -> father
        "'re": "",  # you're -> you are
        "'ve": "",  # I've -> I have
        "'ll": "",  # you'll -> you will
        "'d": "",  # he'd -> he would
        "n't": "",  # can't -> cannot
        "'m": "",  # I'm -> I am
        "'t": "",  # doesn't -> does not (extra handling required to avoid overlap with "n't")
    }

    for contraction, replacement in contractions_dict.items():
        text = re.sub(rf"{contraction}\b", replacement, text)
    return text


def british_to_american(text: str) -> str:
    """Convert British English spellings to American English.

    Args:
        text: The input text in British English.

    Returns:
        The converted text in American English.
    """
    british_to_american_mapping = {
        "colour": "color",
        "honour": "honor",
        "favour": "favor",
        "neighbour": "neighbor",
        "centre": "center",
        "theatre": "theater",
        "metre": "meter",
        "litre": "liter",
        "programme": "program",
        "dialogue": "dialog",
        "analyse": "analyze",
        "realise": "realize",
        "organise": "organize",
        "travelled": "traveled",
        "cancelled": "canceled",
        "modelled": "modeled",
        "labelled": "labeled",
        "signalled": "signaled",
        "marvelled": "marveled",
        "travel": "travel",
        "cancel": "cancel",
        "model": "model",
        "label": "label",
        "signal": "signal",
        "marvel": "marvel",
        "ise": "ize",
        "yse": "yze",
        "our": "or",
        "re": "er",
    }

    words = text.split()
    for i in range(len(words)):
        word = words[i]
        if word.lower() in british_to_american_mapping:
            words[i] = british_to_american_mapping[word.lower()]

    return " ".join(words)


def get_wordnet_pos(tag: str) -> Optional[str]:
    """Convert POS tag to WordNet POS tag.

    Args:
        tag (str): POS tag.

    Raises:
        ValueError: If the POS tag is invalid.

    Returns:
        str: WordNet POS tag.
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemmatizer(text: str) -> str:
    """Lemmatize the input text.

    Args:
        text: The input text.

    Returns:
        The lemmatized text.
    """
    tokens = word_tokenize(text)
    # POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN  # type: ignore
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
    # Join tokens back to string
    clean_text = " ".join(lemmatized_tokens)
    return clean_text
