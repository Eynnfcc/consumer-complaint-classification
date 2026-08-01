# ==========================================================
# MODEL.PY
# ==========================================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Dense,
    Dropout,
)

from tensorflow.keras.callbacks import EarlyStopping


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
# TOKENIZER + PADDING
# ==========================================================

def tokenize_data(X_train, X_test):

    vocab_size = 30000
    max_length = 240

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )

    print("\nBuilding vocabulary...")

    tokenizer.fit_on_texts(X_train.astype(str))

    X_train_seq = tokenizer.texts_to_sequences(X_train.astype(str))
    X_test_seq = tokenizer.texts_to_sequences(X_test.astype(str))

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    return (
        tokenizer,
        X_train_pad,
        X_test_pad,
        vocab_size,
        max_length
    )


# ==========================================================
# LABEL ENCODING
# ==========================================================

def encode_labels(y_train, y_test):

    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    return encoder, y_train_enc, y_test_enc


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

def get_class_weights(y_train_encoded):

    classes = np.unique(y_train_encoded)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_encoded
    )

    return dict(zip(classes, weights))


# ==========================================================
# BUILD SIMPLERNN
# ==========================================================

def build_rnn(vocab_size, max_length, num_classes):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=128,
            input_length=max_length,
            mask_zero=True
        ),

        SimpleRNN(128),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# BUILD LSTM
# ==========================================================

def build_lstm(vocab_size, max_length, num_classes):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=100,
            input_length=max_length,
            mask_zero=True
        ),

        LSTM(128),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# BUILD GRU
# ==========================================================

def build_gru(vocab_size, max_length, num_classes):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=100,
            input_length=max_length,
            mask_zero=True
        ),

        GRU(128),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model,
    X_train,
    y_train,
    class_weights,
):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )

    return history