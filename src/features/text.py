"""Text feature engineering for search queries, titles, and tags."""

import re
import unicodedata

import pandas as pd

_PUNCTUATION_PATTERN = re.compile(r"[,.:;\-_/\\+=()\[\]<>^~?!#*%]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Dropped only from word-frequency analyses, never from the word-count/length features below.
PORTUGUESE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "ao",
        "aos",
        "aquela",
        "aquelas",
        "aquele",
        "aqueles",
        "aquilo",
        "as",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "delas",
        "dele",
        "deles",
        "depois",
        "do",
        "dos",
        "e",
        "é",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estas",
        "este",
        "estes",
        "eu",
        "foi",
        "for",
        "isso",
        "isto",
        "já",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "muito",
        "na",
        "nas",
        "não",
        "nem",
        "no",
        "nos",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "num",
        "numa",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "quem",
        "são",
        "se",
        "sem",
        "ser",
        "seu",
        "seus",
        "só",
        "sua",
        "suas",
        "também",
        "te",
        "tu",
        "tua",
        "tuas",
        "um",
        "uma",
        "umas",
        "uns",
        "você",
        "vocês",
    }
)


def strip_accents(text: str) -> str:
    """Remove accents/diacritics, e.g. 'á' -> 'a'.

    Parameters
    ----------
    text : str
        Text to strip accents from.

    Returns
    -------
    str
        `text` with combining diacritical marks removed.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def clean_text(text: str | float) -> str:
    """Lowercase, strip accents, and replace punctuation with whitespace.

    Parameters
    ----------
    text : str or float
        Raw text to clean. A float is only valid if it is NaN (e.g. a
        missing value in a pandas column), which is treated as an empty
        string.

    Returns
    -------
    str
        The cleaned text: lowercase, accent-free, punctuation replaced by
        single spaces, and stripped of leading/trailing whitespace.
    """
    text = "" if pd.isna(text) else str(text)
    text = strip_accents(text).lower()
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(
    text: str | float, drop_numeric: bool = True, drop_stopwords: bool = False
) -> list[str]:
    """Split cleaned text into tokens.

    Parameters
    ----------
    text : str or float
        Raw text to tokenize; see `clean_text`.
    drop_numeric : bool, default True
        Remove tokens that are purely digits (e.g. "15", "2020"), common in
        tag lists ("15 anos") but rarely informative on their own.
    drop_stopwords : bool, default False
        Remove `PORTUGUESE_STOPWORDS`. Meant for word-frequency analyses,
        not for the word-count/length features below, which should count
        every token.

    Returns
    -------
    list of str
        Cleaned, whitespace-split tokens.
    """
    tokens = clean_text(text).split()
    if drop_numeric:
        tokens = [token for token in tokens if not token.isdigit()]
    if drop_stopwords:
        tokens = [token for token in tokens if token not in PORTUGUESE_STOPWORDS]
    return tokens


def word_count(text: str | float, drop_numeric: bool = True) -> int:
    """Number of tokens in text after cleaning.

    Parameters
    ----------
    text : str or float
        Raw text to count; see `clean_text`.
    drop_numeric : bool, default True
        Passed through to `tokenize`.

    Returns
    -------
    int
        Number of tokens.
    """
    return len(tokenize(text, drop_numeric=drop_numeric))


def char_length(text: str | float) -> int:
    """Number of characters in text after cleaning.

    Parameters
    ----------
    text : str or float
        Raw text to measure; see `clean_text`.

    Returns
    -------
    int
        Length, in characters, of the cleaned text.
    """
    return len(clean_text(text))
