# ==========================================================
# UTILS_TRANSFORMER.PY
# ==========================================================

import numpy as np
from sklearn.metrics import classification_report


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(trainer, test_dataset, label_encoder):

    print("\n" + "=" * 50)
    print("Evaluating Transformer")
    print("=" * 50)

    # Predict on test dataset
    predictions = trainer.predict(test_dataset)

    # Predicted classes
    y_pred = np.argmax(predictions.predictions, axis=1)

    # True labels
    y_true = predictions.label_ids

    # Classification Report
    print("\nClassification Report:\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
    )

    print(report)

    return report