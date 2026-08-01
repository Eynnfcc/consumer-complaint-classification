# ==========================================================
# TRANSFORMER_PREPROCESSING.PY
# ==========================================================

import warnings

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(file_path):

    df = pd.read_csv("complaints_processed.csv")

    print("Dataset Shape:", df.shape)

    return df


# ==========================================================
# PREPROCESS DATASET
# ==========================================================

def preprocess_dataframe(df):

    # Remove invalid rows
    df = df[df["narrative"] != "name"]

    # Remove missing narratives
    df = df.dropna(subset=["narrative"])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    print("\nPreprocessing Completed Successfully!")

    print(f"Remaining Samples: {len(df)}")

    return df


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def split_data(df):

    X = df["narrative"]
    y = df["product"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 50)
    print("Train/Test Split Completed Successfully")
    print("=" * 50)

    print(f"Total Samples : {len(df)}")
    print(f"Training Set  : {len(X_train)}")
    print(f"Testing Set   : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# LABEL ENCODING
# ==========================================================

def encode_labels(y_train, y_test):

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)

    y_test_encoded = encoder.transform(y_test)

    print("\nLabel Encoding Completed Successfully!")

    return (
        encoder,
        y_train_encoded,
        y_test_encoded,
    )

    for i, label in enumerate(label_encoder.classes_):
        print(i, "->", label)