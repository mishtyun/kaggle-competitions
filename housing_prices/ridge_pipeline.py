from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.linear_model import Ridge

from df_simple_imputer import DataFrameSimpleImputer
from base_data_preprocessor import BaseDataPreprocessor

__all__ = ["make_pipeline"]


def get_pipeline_preprocessors(continuous_columns, categorical_columns):
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            ("scaler", BaseDataPreprocessor(needed_columns=continuous_columns)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="most_frequent")),
            ("onehot_encoder", OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    preprocessors = [
        ("numerical", numerical_pipeline, continuous_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]

    return ColumnTransformer(
        transformers=preprocessors,
    )


def make_pipeline(**kwargs):
    preprocessors = get_pipeline_preprocessors(**kwargs)

    grid_param = {'alpha': range(1, 10)}

    rmse_scorer = make_scorer(
        root_mean_squared_error, greater_is_better=False
    )

    sgd_grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=grid_param,
        scoring=rmse_scorer,
        refit=True,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessors),
            ("sgd_grid", sgd_grid),
        ],
    )

    return pipe