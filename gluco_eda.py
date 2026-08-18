import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from load_data import load_data


sns.set(style="whitegrid")

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(
        _chart_path(filename),
        bbox_inches="tight",
        dpi=100
    )
    plt.close("all")


def run_eda() -> dict:

    data = load_data()

    charts = []
    eda_sections = []

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    print("\n" + "=" * 80)
    print("1. LOAD DATA")
    print("=" * 80)

    print("Dataset Shape:", data.shape)

    print("\nFirst 5 Rows:")
    print(data.head())

    eda_sections.append({
        "number": 1,
        "title": "Load Data",
        "description": "Loads the GlucoTrend dataset and checks its basic shape and first records.",
        "chart": None
    })

    # ============================================================
    # 2. BASIC DATA INFORMATION
    # ============================================================

    print("\n" + "=" * 80)
    print("2. BASIC DATA INFORMATION")
    print("=" * 80)

    print("\nData Types:")
    print(data.dtypes)

    print("\nNumerical Statistics:")
    print(data.describe())

    eda_sections.append({
        "number": 2,
        "title": "Basic Data Information",
        "description": "Examines dataset structure, data types and descriptive statistics.",
        "chart": None
    })

    # ============================================================
    # 3. MISSING VALUES
    # ============================================================

    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)

    missing = data.isnull().sum()

    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        by="missing_count",
        ascending=False
    )

    print(missing_df)

    chart_file = None

    if not missing_df.empty:

        plt.figure(figsize=(12, 5))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel("Missing Percentage")
        plt.xlabel("Column")
        plt.title("Missing Values by Column")

        chart_file = "missing_values.png"

        _save(chart_file)

        charts.append(chart_file)

    eda_sections.append({
        "number": 3,
        "title": "Missing Value Analysis",
        "description": "Identifies columns containing missing values and shows their percentage.",
        "chart": chart_file
    })

    # ============================================================
    # 4. DUPLICATE ROWS
    # ============================================================

    print("\n" + "=" * 80)
    print("4. DUPLICATE ROWS")
    print("=" * 80)

    duplicate_count = int(data.duplicated().sum())

    print("Duplicate Rows:", duplicate_count)

    eda_sections.append({
        "number": 4,
        "title": "Duplicate Row Analysis",
        "description": f"The dataset contains {duplicate_count} duplicate rows.",
        "chart": None
    })

    # ============================================================
    # 5. GLUCOSE DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 80)
    print("5. GLUCOSE DISTRIBUTION")
    print("=" * 80)

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data["glucose"],
        bins=30,
        kde=True
    )

    plt.axvline(
        data["glucose"].mean(),
        linestyle="--",
        label="Mean"
    )

    plt.xlabel("Glucose")
    plt.ylabel("Frequency")
    plt.title("Glucose Distribution")
    plt.legend()

    chart_file = "glucose_distribution.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 5,
        "title": "Glucose Distribution",
        "description": "Analyzes how glucose measurements are distributed across the dataset.",
        "chart": chart_file
    })

    # ============================================================
    # 6. NUMERIC FEATURE DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 80)
    print("6. NUMERIC FEATURE DISTRIBUTION")
    print("=" * 80)

    numeric_cols = [
        "glucose",
        "insulin_bolus",
        "insulin_basal",
        "carbs",
        "exercise_steps",
        "heart_rate",
        "skin_temp",
        "gsr",
        "stress_level",
        "hbA1c",
        "age",
        "weight"
    ]

    numeric_cols = [
        col for col in numeric_cols
        if col in data.columns
    ]

    data[numeric_cols].hist(
        figsize=(15, 12),
        bins=20
    )

    plt.tight_layout()

    chart_file = "numeric_distribution.png"

    plt.savefig(
        _chart_path(chart_file),
        dpi=100
    )

    plt.close("all")

    charts.append(chart_file)

    eda_sections.append({
        "number": 6,
        "title": "Numeric Feature Distribution",
        "description": "Displays distributions of important numerical GlucoTrend features.",
        "chart": chart_file
    })

    # ============================================================
    # 7. OUTLIER DETECTION
    # ============================================================

    print("\n" + "=" * 80)
    print("7. OUTLIER DETECTION")
    print("=" * 80)

    box_cols = [
        "glucose",
        "insulin_bolus",
        "insulin_basal",
        "carbs",
        "exercise_steps",
        "heart_rate",
        "skin_temp",
        "gsr",
        "stress_level",
        "hbA1c",
        "age",
        "weight"
    ]

    box_cols = [
        col for col in box_cols
        if col in data.columns
    ]

    plt.figure(figsize=(15, 6))

    sns.boxplot(
        data=data[box_cols]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.title("Outlier Detection")

    chart_file = "outlier_detection.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 7,
        "title": "Outlier Detection",
        "description": "Uses boxplots to identify possible outliers in numerical health measurements.",
        "chart": chart_file
    })

    # ============================================================
    # 8. CORRELATION ANALYSIS
    # ============================================================

    print("\n" + "=" * 80)
    print("8. CORRELATION ANALYSIS")
    print("=" * 80)

    corr = data.select_dtypes(
        include=[np.number]
    ).corr()

    print(corr)

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        np.round(corr, 2),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    chart_file = "correlation_heatmap.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 8,
        "title": "Correlation Analysis",
        "description": "Examines relationships between numerical GlucoTrend variables.",
        "chart": chart_file
    })

    # ============================================================
    # 9. CARBS VS GLUCOSE
    # ============================================================

    print("\n" + "=" * 80)
    print("9. CARBOHYDRATES VS GLUCOSE")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    sns.regplot(
        x="carbs",
        y="glucose",
        data=data,
        scatter_kws={
            "alpha": 0.5
        }
    )

    plt.xlabel("Carbohydrates")
    plt.ylabel("Glucose")
    plt.title("Carbohydrates vs Glucose")

    chart_file = "carbs_vs_glucose.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 9,
        "title": "Carbohydrates vs Glucose",
        "description": "Analyzes the relationship between carbohydrate intake and glucose levels.",
        "chart": chart_file
    })

    # ============================================================
    # 10. HEART RATE VS GLUCOSE
    # ============================================================

    print("\n" + "=" * 80)
    print("10. HEART RATE VS GLUCOSE")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x="heart_rate",
        y="glucose",
        data=data
    )

    plt.xlabel("Heart Rate")
    plt.ylabel("Glucose")
    plt.title("Heart Rate vs Glucose")

    chart_file = "heart_rate_vs_glucose.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 10,
        "title": "Heart Rate vs Glucose",
        "description": "Examines whether heart rate measurements are related to glucose levels.",
        "chart": chart_file
    })

    # ============================================================
    # 11. MEAL TYPE ANALYSIS
    # ============================================================

    print("\n" + "=" * 80)
    print("11. MEAL TYPE ANALYSIS")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="meal_type",
        data=data
    )

    plt.xticks(rotation=30)

    plt.xlabel("Meal Type")
    plt.ylabel("Count")
    plt.title("Meal Type Distribution")

    chart_file = "meal_type_distribution.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 11,
        "title": "Meal Type Analysis",
        "description": "Shows the frequency of different meal types recorded in the dataset.",
        "chart": chart_file
    })

    # ============================================================
    # 12. SLEEP STAGE ANALYSIS
    # ============================================================

    print("\n" + "=" * 80)
    print("12. SLEEP STAGE ANALYSIS")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="sleep_stage",
        data=data
    )

    plt.xticks(rotation=30)

    plt.xlabel("Sleep Stage")
    plt.ylabel("Count")
    plt.title("Sleep Stage Distribution")

    chart_file = "sleep_stage_distribution.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 12,
        "title": "Sleep Stage Analysis",
        "description": "Examines the distribution of recorded sleep stages.",
        "chart": chart_file
    })

    # ============================================================
    # 13. STRESS LEVEL ANALYSIS
    # ============================================================

    print("\n" + "=" * 80)
    print("13. STRESS LEVEL ANALYSIS")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data["stress_level"],
        bins=15,
        kde=True
    )

    plt.xlabel("Stress Level")
    plt.ylabel("Frequency")
    plt.title("Stress Level Distribution")

    chart_file = "stress_level_distribution.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 13,
        "title": "Stress Level Analysis",
        "description": "Analyzes the distribution of stress levels recorded in the dataset.",
        "chart": chart_file
    })

    # ============================================================
    # 14. GLUCOSE TREND OVER TIME
    # ============================================================

    print("\n" + "=" * 80)
    print("14. GLUCOSE TREND OVER TIME")
    print("=" * 80)

    temp = data.copy()

    temp["timestamp"] = pd.to_datetime(
        temp["timestamp"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=["timestamp"]
    )

    temp = temp.sort_values(
        "timestamp"
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        temp["timestamp"],
        temp["glucose"]
    )

    plt.xlabel("Time")
    plt.ylabel("Glucose")
    plt.title("Glucose Trend Over Time")

    plt.xticks(rotation=45)

    chart_file = "glucose_trend.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 14,
        "title": "Glucose Trend Over Time",
        "description": "Shows how glucose measurements change over time.",
        "chart": chart_file
    })

    # ============================================================
    # 15. EXERCISE INTENSITY VS GLUCOSE
    # ============================================================

    print("\n" + "=" * 80)
    print("15. EXERCISE INTENSITY VS GLUCOSE")
    print("=" * 80)

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        x="exercise_intensity",
        y="glucose",
        data=data
    )

    plt.xlabel("Exercise Intensity")
    plt.ylabel("Glucose")
    plt.title("Exercise Intensity vs Glucose")

    plt.xticks(rotation=30)

    chart_file = "exercise_intensity_vs_glucose.png"

    _save(chart_file)
    charts.append(chart_file)

    eda_sections.append({
        "number": 15,
        "title": "Exercise Intensity vs Glucose",
        "description": "Compares glucose levels across different exercise intensity levels.",
        "chart": chart_file
    })

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return {
        "n_rows": data.shape[0],
        "n_cols": data.shape[1],
        "duplicate_count": duplicate_count,
        "missing": missing.to_dict(),
        "charts": charts,
        "eda_sections": eda_sections
    }