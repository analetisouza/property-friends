import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from exceptions import EmptyPath, DataNotLoaded, ModelNotTrained

CATEGORICAL_COLUMNS = ["type", "sector"]
TARGET_COLUMN = "price"
COLUMNS_TO_IGNORE = ['id', 'target']

LEARNING_RATE = 0.01
NUMBER_OF_ESTIMATORS = 300
MAXIMUM_DEPTH = 5
LOSS = "absolute_error"


class Loader:
    """
    Loads the data for training and testing.
    """
    def __init__(self, train_path: str, test_path: str):
        """
        Validates and initializes data from file paths.
        :param train_path: File path for train data
        :param test_path: File path test data
        """
        self.train, self.test = Loader._get_train_test_dataframe(train_path, test_path)

    @property
    def get_train_columns(self) -> list[str]:
        """
        Excludes the columns to be ignored from the total list of columns on the train dataset.
        :return: List of columns for training
        """
        return [col for col in self.train.columns if col not in COLUMNS_TO_IGNORE]

    @staticmethod
    def _validate_paths(train_path: any, test_path: any):
        """
        Validates if the file paths are not an empty string or None.
        :param train_path: File path for train data csv
        :param test_path: File path for test data csv
        """
        if not train_path:
            raise EmptyPath("train_path")
        if not test_path:
            raise EmptyPath("test_path")

    @staticmethod
    def _get_train_test_dataframe(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validates and initializes data from csv files.
        :param train_path: File path for train data csv
        :param test_path: File path test data
        :return: Two dataframes for training and testing
        """
        Loader._validate_paths(train_path, test_path)
        return pd.read_csv(train_path), pd.read_csv(test_path)


class Trainer:
    """
    Creates a pipeline of steps to train the model and runs it.
    """
    def __init__(self, loader: Loader):
        """
        Initializes pipeline and trains model.
        :param loader: Loaded data
        """
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
        self.train_model(loader)

    def train_model(self, loader: Loader):
        """
        Fits the model with the data provided from the class Loader.
        :param loader: Loaded data
        """
        Trainer._validate_loading(loader)
        self.pipeline.fit(loader.train[loader.get_train_columns], loader.train[TARGET_COLUMN])

    @staticmethod
    def _validate_loading(loader: any):
        """
        Validates if the class Loader is initialized.
        :param loader: Loaded data
        """
        if not isinstance(loader, Loader):
            raise DataNotLoaded()


class Evaluator:
    """
    Evaluates the model's predictions accuracy.
    """
    def __init__(self, loader: Loader, trainer: Trainer):
        """
        Validates training, makes prediction using test data and calculates metrics.
        :param loader: Loaded data
        :param trainer: Trained model
        """
        Evaluator._validate_training(trainer)
        self.predictions = trainer.pipeline.predict(loader.test[loader.get_train_columns])
        self.target = loader.test[TARGET_COLUMN].values
        self.rmse = np.sqrt(mean_squared_error(self.predictions, self.target))
        self.mape = mean_absolute_percentage_error(self.predictions, self.target)
        self.mae = mean_absolute_error(self.predictions, self.target)

    def print_metrics(self):
        """
        Prints the model's metrics.
        """
        print("RMSE: ", self.rmse)
        print("MAPE: ", self.mape)
        print("MAE : ", self.mae)

    @staticmethod
    def _validate_training(trainer: any):
        """
        Validates if the class Trainer is initialized.
        :param trainer: Trained model
        """
        if not isinstance(trainer, Trainer):
            raise ModelNotTrained()
