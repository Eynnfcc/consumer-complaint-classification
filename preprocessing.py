# ==========================================================
# PREPROCESSING.PY
# ==========================================================

import re
import string
import warnings
import contractions
import spacy
import nltk

from tqdm import tqdm
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

tqdm.pandas()

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("words", quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load stopwords
stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stop_words.discard("never")


# ==========================================================
# TEXT CLEANING FUNCTIONS
# ==========================================================

def expand_contractions(text):
    if isinstance(text, str):
        return contractions.fix(text)
    return text


def remove_punctuation(text):
    if not isinstance(text, str):
        return text
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emails(text):
    return re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)


def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub("", text)


def lemmatize_text(text):
    doc = nlp(text)
    lemmas = []

    for token in doc:
        if not token.is_space:
            lemmas.append(token.lemma_)

    return " ".join(lemmas)


def remove_stopwords(text):
    if not isinstance(text, str):
        return ""

    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)


# ==========================================================
# COMPLETE PREPROCESSING PIPELINE
# ==========================================================

def preprocess_dataframe(df):

    print("\nExpanding contractions...")
    df["narrative"] = df["narrative"].progress_apply(expand_contractions)

    print("\nConverting to lowercase...")
    df["narrative"] = df["narrative"].str.lower()

    print("\nRemoving punctuation...")
    df["narrative"] = (
        df["narrative"]
        .fillna("")
        .progress_apply(remove_punctuation)
    )

    print("\nRemoving URLs...")
    df["narrative"] = df["narrative"].progress_apply(remove_urls)

    print("Removing Emails...")
    df["narrative"] = df["narrative"].progress_apply(remove_emails)

    print("Removing Mentions...")
    df["narrative"] = df["narrative"].progress_apply(remove_mentions)

    print("Removing Hashtags...")
    df["narrative"] = df["narrative"].progress_apply(remove_hashtags)

    print("Removing Emojis...")
    df["narrative"] = df["narrative"].progress_apply(remove_emojis)

    print("\nLemmatizing text...")
    df["narrative"] = df["narrative"].progress_apply(lemmatize_text)

    print("\nRemoving Stopwords...")
    df["narrative"] = df["narrative"].progress_apply(remove_stopwords)

    return df