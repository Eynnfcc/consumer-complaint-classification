# ==========================================================
# TRANSFORMER_MAIN.PY
# ==========================================================

import warnings
import torch

from transformer_preprocessing import (
    load_dataset,
    preprocess_dataframe,
    split_data,
    encode_labels,
)

from transformer_model import (
    load_tokenizer,
    tokenize_data,
    build_model,
    train_model,
)

from transformer_utils import (
    evaluate_model,
)

warnings.filterwarnings("ignore")

print("✅ Libraries Loaded Successfully!")

# ==========================================================
# LOAD DATASET
# ==========================================================

file_path = "complaints_processed.csv"

df = load_dataset(file_path)

# ==========================================================
# PREPROCESS DATASET
# ==========================================================

df = preprocess_dataframe(df)

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = split_data(df)

# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder, y_train_enc, y_test_enc = encode_labels(
    y_train,
    y_test,
)
print("\nLabel Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")
# ==========================================================
# LOAD TOKENIZER
# ==========================================================

tokenizer = load_tokenizer()

# ==========================================================
# TOKENIZE DATA
# ==========================================================

train_dataset, test_dataset = tokenize_data(
    tokenizer,
    X_train,
    X_test,
    y_train_enc,
    y_test_enc,
)

print(train_dataset[0])
print(train_dataset.features)

# ==========================================================
# BUILD MODEL
# ==========================================================

model = build_model(
    num_classes=len(label_encoder.classes_)
)
print("Model device:", next(model.parameters()).device)
print("CUDA Available:", torch.cuda.is_available())
# ==========================================================
# TRAIN MODEL
# ==========================================================

trainer = train_model(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    tokenizer=tokenizer,
)

# ==========================================================
# EVALUATE MODEL
# ==========================================================

evaluate_model(
    trainer=trainer,
    test_dataset=test_dataset,
    label_encoder=label_encoder,
)

print("\n✅ Transformer Project Finished Successfully!")