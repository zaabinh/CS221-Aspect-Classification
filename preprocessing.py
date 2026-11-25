"""Inference-time copy of the text pipeline in notebooks/01-preprocessing.ipynb.

clean_review_text  -> the dataset's `review_en` column
preprocess_review  -> the dataset's `review_cleaned` column (what the classifiers were trained on)
"""
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import contractions
except ImportError:
    contractions = None

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError as error:
    raise RuntimeError(
        "Missing NLTK stopwords data. Run: python -m nltk.downloader stopwords wordnet omw-1.4"
    ) from error

# Negations are kept, exactly as in the preprocessing notebook.
NEGATION_WORDS = {"no", "nor", "not", "never", "without", "cannot"}
STOP_WORDS_FOR_ML = STOP_WORDS - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    if contractions is not None:
        return contractions.fix(text)

    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for contraction, expansion in replacements.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def clean_review_text(review: str) -> str:
    if not isinstance(review, str):
        raise TypeError("Review must be a string")

    review = review.replace("​", " ")
    review = _expand_contractions(review)
    review = review.lower()
    review = re.sub(r"[^a-z0-9\s]", " ", review)
    return re.sub(r"\s+", " ", review).strip()


def preprocess_review(review: str) -> str:
    review = clean_review_text(review)
    if not review:
        raise ValueError("Review cannot be empty")

    return " ".join(
        LEMMATIZER.lemmatize(word)
        for word in review.split()
        if word not in STOP_WORDS_FOR_ML
    )
