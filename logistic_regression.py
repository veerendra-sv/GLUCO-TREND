import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from load_data import load_data


def run_logistic_regression():
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
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Normal Glucose (<140)", "High Glucose (>=140)"],
            output_dict=True
        )

        return {
            "model": "Logistic Regression Classifier",
            "features": features,
            "target": "High Glucose Risk (>=140 mg/dL)",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "accuracy": round(accuracy * 100, 2),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "classes": ["Normal Glucose (<140)", "High Glucose (>=140)"]
        }

    except Exception as e:
        return {
            "error": str(e)
        }
