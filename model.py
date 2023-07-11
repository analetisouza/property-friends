import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# TRAIN_FILE_NAME = 'train.csv'
# TEST_FILE_NAME = 'test.csv'
CATEGORICAL_COLUMNS = ["type", "sector"]
TARGET_COLUMN = "price"
COLUMNS_TO_IGNORE = ['id', 'target']

LEARNING_RATE = 0.01
NUMBER_OF_ESTIMATORS = 300
MAXIMUM_DEPTH = 5
LOSS = "absolute_error"


class Loader:
    def __init__(self, train_path: str, test_path: str):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    @property
    def train_columns(self):
        return [col for col in self.train.columns if col not in COLUMNS_TO_IGNORE]


class Trainer:
    def __init__(self):
        self.categorical_transformer = TargetEncoder()
        self.preprocessor = ColumnTransformer(transformers=[('categorical',
                                                             self.categorical_transformer,
                                                             CATEGORICAL_COLUMNS)])

        self.steps = [('preprocessor', self.preprocessor),
                      ('model', GradientBoostingRegressor(**{"learning_rate": LEARNING_RATE,
                                                             "n_estimators": NUMBER_OF_ESTIMATORS,
                                                             "max_depth": MAXIMUM_DEPTH,
                                                             "loss": LOSS,
                                                             }))]
        self.pipeline = Pipeline(self.steps)

    def train_model(self, loader: Loader):
        self.pipeline.fit(loader.train[loader.train_columns], loader.train[TARGET_COLUMN])


class Evaluator:
    def __init__(self, loader: Loader, trainer: Trainer):
        self.predictions = trainer.pipeline.predict(loader.test[loader.train_columns])
        self.target = loader.test[TARGET_COLUMN].values
        self.rmse = np.sqrt(mean_squared_error(self.predictions, self.target))
        self.mape = mean_absolute_percentage_error(self.predictions, self.target)
        self.mae = mean_absolute_error(self.predictions, self.target)

    def print_metrics(self):
        print("RMSE: ", self.rmse)
        print("MAPE: ", self.mape)
        print("MAE : ", self.mae)
