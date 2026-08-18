import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "GlucoBench_benchmark_dataset.csv"
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset file does not exist: {path}"
        )

    df = pd.read_csv(path)

    return df


def get_data_summary(path: str = DATA_PATH) -> dict:

    df = load_data(path)

    summary = {

        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict(
            orient="records"
        )

    }

    return summary


if __name__ == "__main__":

    summary = get_data_summary()

    print(summary)