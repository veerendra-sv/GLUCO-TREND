import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from load_data import load_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def standardize(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std()
    if std == 0 or not np.isfinite(std):
        std = 1.0
    return (x - mean) / std, mean, std


def batch_gradient_descent(x, y, alpha=0.05, epochs=10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(x)

    theta0 = 0.0
    theta1 = 0.0
    mse_history = []

    for epoch in range(1, epochs + 1):
        y_hat = theta0 + theta1 * x
        error = y_hat - y

        grad_theta0 = (2.0 / m) * np.sum(error)
        grad_theta1 = (2.0 / m) * np.sum(error * x)

        theta0 = theta0 - alpha * grad_theta0
        theta1 = theta1 - alpha * grad_theta1

        updated_y_hat = theta0 + theta1 * x
        mse = np.mean((updated_y_hat - y) ** 2)

        mse_history.append({
            "epoch": epoch,
            "theta0": float(round(theta0, 4)),
            "theta1": float(round(theta1, 4)),
            "mse": float(round(mse, 4))
        })

    return theta0, theta1, mse_history


def charts_directory():
    chart_dir = os.path.join(BASE_DIR, "static", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir


def save_figure(fig, filename):
    chart_dir = charts_directory()
    path = os.path.join(chart_dir, filename)
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def run_linear_regression():
    try:
        df = load_data()

        feature_cols = [
            "insulin_bolus", "insulin_basal", "carbs",
            "exercise_steps", "heart_rate", "skin_temp",
            "gsr", "stress_level", "hbA1c", "age",
            "weight", "carb_ratio", "insulin_sensitivity"
        ]

        target_col = "glucose"

        available_features = [col for col in feature_cols if col in df.columns]
        data = df[available_features + [target_col]].dropna()

        for col in available_features + [target_col]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna()

        X = data[available_features]
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Scikit-Learn Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Single feature Gradient Descent demonstration
        primary_feature = "hbA1c" if "hbA1c" in available_features else available_features[0]
        x_single = data[primary_feature].values
        y_single = data[target_col].values

        x_scaled, mean_x, std_x = standardize(x_single)

        theta0, theta1, mse_history = batch_gradient_descent(
            x_scaled, y_single, alpha=0.05, epochs=10
        )

        # Generate charts
        chart_dir = charts_directory()

        # 1. MSE History Chart
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        epochs = [item["epoch"] for item in mse_history]
        mses = [item["mse"] for item in mse_history]
        ax1.plot(epochs, mses, marker="o", color="#00bcd4", linewidth=2)
        ax1.set_title("Batch Gradient Descent - Loss Curve (MSE)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Squared Error")
        ax1.grid(True, linestyle="--", alpha=0.5)
        mse_chart_path = save_figure(fig1, "gradient_descent_mse.png")

        # 2. Actual vs Predicted Chart
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sample_indices = np.random.choice(len(y_test), min(200, len(y_test)), replace=False)
        ax2.scatter(y_test.iloc[sample_indices], y_pred[sample_indices], alpha=0.6, color="#e91e63", edgecolors="k")
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")
        ax2.set_title("Actual vs Predicted Blood Glucose (mg/dL)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Actual Glucose")
        ax2.set_ylabel("Predicted Glucose")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        pred_chart_path = save_figure(fig2, "linear_regression_actual_vs_pred.png")

        coefficients = dict(zip(available_features, np.round(model.coef_, 4)))

        return {
            "model": "Linear Regression & Batch Gradient Descent",
            "target": target_col,
            "features": available_features,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": {
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2_score": round(r2, 4)
            },
            "intercept": round(float(model.intercept_), 4),
            "coefficients": coefficients,
            "gradient_descent": {
                "feature": primary_feature,
                "epochs": mse_history,
                "final_theta0": round(theta0, 4),
                "final_theta1": round(theta1, 4)
            },
            "charts": {
                "mse_history": mse_chart_path,
                "actual_vs_pred": pred_chart_path
            }
        }

    except Exception as e:
        return {
            "error": str(e)
        }
