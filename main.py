# ==========================================================
# MAIN.PY
# ==========================================================

import warnings
import pandas as pd

from preprocessing import preprocess_dataframe

from model import (
    split_data,
    tokenize_data,
    encode_labels,
    get_class_weights,
    build_rnn,
    build_lstm,
    build_gru,
    train_model,
)

from utils import evaluate_model

warnings.filterwarnings("ignore")

print("✅ Libraries Loaded Successfully!")

# ==========================================================
# LOAD DATASET
# ==========================================================

file_path = "complaints_processed.csv"

df_original = pd.read_csv(file_path)

# Remove invalid rows
df = df_original[df_original["narrative"] != "name"]

# Reset index
df.reset_index(drop=True, inplace=True)

print("Dataset Shape:", df.shape)

# ==========================================================
# PREPROCESSING
# ==========================================================

df = preprocess_dataframe(df)

print("\nPreprocessing Completed Successfully!")

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = split_data(df)

# ==========================================================
# TOKENIZER + PADDING
# ==========================================================

tokenizer, X_train_pad, X_test_pad, vocab_size, max_length = tokenize_data(
    X_train,
    X_test,
)

# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder, y_train_enc, y_test_enc = encode_labels(
    y_train,
    y_test,
)

# ==========================================================
# CLASS WEIGHTS
# ==========================================================

class_weights = get_class_weights(y_train_enc)

# ==========================================================
# BUILD MODELS
# ==========================================================

models = {
    "SimpleRNN": build_rnn(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=len(label_encoder.classes_),
    ),

    "LSTM": build_lstm(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=len(label_encoder.classes_),
    ),

    "GRU": build_gru(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=len(label_encoder.classes_),
    ),
}

# ==========================================================
# TRAIN & EVALUATE
# ==========================================================

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training {name}")
    print("=" * 70)
    model.build(input_shape=(None, max_length))

    model.summary()

    history = train_model(
        model=model,
        X_train=X_train_pad,
        y_train=y_train_enc,
        class_weights=class_weights,
    )

    evaluate_model(
        model=model,
        X_test=X_test_pad,
        y_test=y_test_enc,
        label_encoder=label_encoder,
    )

print("\n✅ Project Finished Successfully!")