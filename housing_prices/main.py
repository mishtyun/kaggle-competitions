import pandas as pd

from utils import load_data, split_data, split_feature_columns
from ridge_pipeline import make_pipeline as make_ridge_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator


def test_ridge_pipeline(save_to_csv: bool = False):
    train, test = load_data()
    target_column = "SalePrice"

    data_train, data_test, Y_train, Y_test = split_data(train, target_column)

    continuous_columns, categorical_columns = split_feature_columns(data_train)


    X_train = data_train[continuous_columns + categorical_columns]
    X_test = data_test[continuous_columns + categorical_columns]

    pipe = make_ridge_pipeline(continuous_columns=continuous_columns, categorical_columns=categorical_columns)
    pipe.fit(X_train, Y_train)
    pipe_pred = pipe.predict(X_test)

    pipe_ridge__rmse = root_mean_squared_error(Y_test, pipe_pred)
    print(f"Ridge RMSE : {pipe_ridge__rmse}\n")
    
    if save_to_csv:
        save_predict_to_csv(pipe, test, target_column)

    return pipe, test


def save_predict_to_csv(model_to_test: BaseEstimator, df_to_predict: pd.DataFrame, target_column: str, output_file_name: str = "output.csv"):
    pipe_real_pred = model_to_test.predict(df_to_predict)

    df = pd.DataFrame(pipe_real_pred, columns=[target_column])

    df.insert(0, "Id", df_to_predict.Id.values)
    df.to_csv(output_file_name, index=False)


if __name__ == '__main__':
    test_ridge_pipeline(True)  # kaggle score: 17698.47