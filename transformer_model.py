# ==========================================================
# TRANSFORMER_MODEL.PY
# ==========================================================

import numpy as np

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ==========================================================
# MODEL NAME
# ==========================================================      


MODEL_NAME = "roberta-base"


# ==========================================================
# LOAD TOKENIZER
# ==========================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizer Loaded Successfully!")

    return tokenizer


# ==========================================================
# TOKENIZE DATA
# ==========================================================

def tokenize_data(
    tokenizer,
    X_train,
    X_test,
    y_train,
    y_test,
):

    train_dataset = Dataset.from_dict({
    "text": X_train.tolist(),
    "label": y_train.tolist(),
    })

    test_dataset = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist(),
    })

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)

    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label",
        ],
    )

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label",
        ],
    )

    print("Tokenization Completed Successfully!")

    return train_dataset, test_dataset


# ==========================================================
# LOAD MODEL
# ==========================================================

def build_model(num_classes):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
    )

    print("Model Loaded Successfully!")

    return model


# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model,
    train_dataset,
    test_dataset,
    tokenizer,
):

    training_args = TrainingArguments(

        output_dir="./results",

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        load_best_model_at_end=True,

        logging_dir="./logs",

        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics,
    )


    
    trainer.train()
    sample = tokenizer(
            "My credit card was charged twice.",
            return_tensors="pt"
        )

    model.eval()

    with torch.no_grad():
        outputs = model(**sample)

    print(outputs.logits)
    
    trainer.save_model("./saved_model")
    tokenizer.save_pretrained("./saved_model")


    return trainer