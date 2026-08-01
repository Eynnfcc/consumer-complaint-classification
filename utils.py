# ==========================================================
# UTILS.PY
# ==========================================================

import numpy as np
from sklearn.metrics import classification_report


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model, X_test, y_test, label_encoder, model_name):

    print("\n" + "=" * 50)
    print(f"Evaluating {model_name}")
    print("=" * 50)

    # Predict
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )

    print(report)

    return report