from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

SEED = 24

__all__ = ["load_data", "split_data", "split_feature_columns"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = "housing_prices/data/"

    test = pd.read_csv(base + "test.csv")
    train = pd.read_csv(base + "train.csv")

    return train, test

def split_data(train: pd.DataFrame, target_column: str, use_standard_seed: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_standard_seed:
        np.random.seed(SEED)

    test_size = 0.2
    data_train, data_test, Y_train, Y_test = train_test_split(
        train[train.columns.drop(target_column)],
        np.array(train[target_column]),
        test_size=test_size,
        random_state=SEED,
    )
    return data_train, data_test, Y_train, Y_test

def split_feature_columns(
    data: pd.DataFrame, target_column: str | None = None
) -> tuple[list, list]:
    continuous_columns = [
        key for key in data.keys() if data[key].dtype in ("int64", "float64")
    ]
    categorical_columns = [key for key in data.keys() if data[key].dtype == "object"]

    if target_column:
        continuous_columns.remove(target_column)

    print(
        f"Continuous : {len(continuous_columns)}, Categorical : {len(categorical_columns)}"
    )
    return continuous_columns, categorical_columns