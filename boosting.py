import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from load_data import load_data


def run_boosting():
    try:
        df = load_data()

        feature_cols = [
            "insulin_bolus", "insulin_basal", "carbs",
            "exercise_steps", "heart_rate", "skin_temp",
            "gsr", "stress_level", "hbA1c", "age",
            "weight", "carb_ratio", "insulin_sensitivity"
        ]

        features = [col for col in feature_cols if col in df.columns]

        data = df[features + ["glucose"]].dropna().copy()

        for col in features + ["glucose"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna()

        # Binary target: High Glucose (>= 140 mg/dL)
        data["high_glucose"] = (data["glucose"] >= 140).astype(int)

        X = data[features]
        y = data["high_glucose"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        feature_importance = dict(zip(features, np.round(model.feature_importances_, 4)))

        return {
            "model": "Gradient Boosting Classifier",
            "target": "High Glucose Risk (>=140 mg/dL)",
            "features": features,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "feature_importance": feature_importance
        }

    except Exception as e:
        return {
            "error": str(e)
        }
