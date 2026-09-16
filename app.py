from flask import Flask, render_template
import traceback

from load_data import load_data, get_data_summary, DATA_PATH
from Mini_Max_Scaling import run_minmax_scaling
from one_hot_encoding import run_onehot_encoding
from Ordinal_Encoding import run_ordinal_encoding
from Standard_Scaler import run_standard_scaling

import gluco_eda

from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from bagging import run_bagging
from boosting import run_boosting


app = Flask(__name__)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )


# ==========================================================
# DATA LOADING
# ==========================================================

@app.route("/data-loading")
def data_loading():
    try:
        print("\n" + "=" * 70)
        print("                 LOADING GLUCO TREND DATASET")
        print("=" * 70)
        print("\nDataset Path:", DATA_PATH)

        summary = get_data_summary()

        print("\nDataset loaded successfully!")
        print("Dataset Shape:", (summary["n_rows"], summary["n_cols"]))

        return render_template(
            "data_loading.html",
            active="data-loading",
            summary=summary
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "data_loading.html",
            active="data-loading",
            error=str(e)
        )


# ==========================================================
# EDA
# ==========================================================

@app.route("/eda")
def eda_page():
    try:
        results = gluco_eda.run_eda()
        return render_template(
            "eda.html",
            active="eda",
            results=results
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "eda.html",
            active="eda",
            error=str(e)
        )


# ==========================================================
# PREPROCESSING
# ==========================================================

@app.route("/preprocessing")
def preprocessing():
    try:
        print("\n" + "=" * 70)
        print("          GLUCO TREND PREPROCESSING PIPELINE")
        print("=" * 70)

        df = load_data()

        train_df, test_df, minmax_scaler = run_minmax_scaling(df)
        train_df, test_df, ohe = run_onehot_encoding(train_df, test_df)
        train_df, test_df, ordinal_encoder = run_ordinal_encoding(train_df, test_df)
        train_df, test_df, standard_scaler = run_standard_scaling(train_df, test_df)

        target_column = "glucose"
        if target_column not in train_df.columns:
            raise ValueError("Target column 'glucose' was not found in the dataset.")

        y_train = train_df[target_column].copy()
        y_test = test_df[target_column].copy()

        columns_to_remove = [
            "glucose", "timestamp", "user_id", "glucose_roll_mean_1h"
        ]

        X_train = train_df.drop(columns=columns_to_remove, errors="ignore")
        X_test = test_df.drop(columns=columns_to_remove, errors="ignore")

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            success=True,
            original_shape=df.shape,
            train_shape=X_train.shape,
            test_shape=X_test.shape,
            target_train_shape=y_train.shape,
            target_test_shape=y_test.shape,
            feature_count=X_train.shape[1],
            train_preview=X_train.head(10).to_html(classes="data-table", index=False),
            test_preview=X_test.head(10).to_html(classes="data-table", index=False)
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            "preprocessing.html",
            active="preprocessing",
            success=False,
            error=str(e)
        )


# ==========================================================
# LINEAR REGRESSION
# ==========================================================

@app.route("/linear-regression")
def linear_regression():
    try:
        result = run_linear_regression()
        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=None,
            error=str(e)
        )


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

@app.route("/logistic-regression")
def logistic_regression():
    try:
        result = run_logistic_regression()
        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=None,
            error=str(e)
        )


# ==========================================================
# DECISION TREE
# ==========================================================

@app.route("/decision-tree")
def decision_tree():
    try:
        result = run_decision_tree()
        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=None,
            error=str(e)
        )


# ==========================================================
# RANDOM FOREST
# ==========================================================

@app.route("/random-forest")
def random_forest():
    try:
        result = run_random_forest()
        return render_template(
            "random_forest.html",
            active="random-forest",
            result=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "random_forest.html",
            active="random-forest",
            result=None,
            error=str(e)
        )


# ==========================================================
# BAGGING
# ==========================================================

@app.route("/bagging")
def bagging():
    try:
        result = run_bagging()
        return render_template(
            "bagging.html",
            active="bagging",
            result=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "bagging.html",
            active="bagging",
            result=None,
            error=str(e)
        )


# ==========================================================
# BOOSTING
# ==========================================================

@app.route("/boosting")
def boosting():
    try:
        result = run_boosting()
        return render_template(
            "boosting.html",
            active="boosting",
            result=result,
            error=result.get("error")
        )
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "boosting.html",
            active="boosting",
            result=None,
            error=str(e)
        )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )