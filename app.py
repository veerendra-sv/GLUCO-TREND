from flask import Flask, render_template
from load_data import get_data_summary
from gluco_eda import run_eda
import traceback

app = Flask(__name__)


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:
        summary = get_data_summary()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:
        results = run_eda()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=results,
        error=error
    )


# ---------------------------------------------------------
# Run Flask
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)