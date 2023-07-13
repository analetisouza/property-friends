import pytest
from model import Loader, Trainer, Evaluator
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from exceptions import EmptyPath, DataNotLoaded, ModelNotTrained

TRAIN_FILE_PATH = 'train.csv'
TEST_FILE_PATH = 'test.csv'
FEATURES = ['type', 'sector', 'net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'latitude', 'longitude', 'price']
FEATURE_TYPES = ['object', 'object', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64']
NUMBER_OF_FEATURES = 8
MODEL_TYPE = 'regressor'
RMSE_THRESHOLD = 11000.0
MAPE_THRESHOLD = 0.5
MAE_THRESHOLD = 6000.0
TARGET_TYPE = ('int64', 'float64')


def test_loader():
    """
    Tests the Loader class.
    """
    loader = Loader(TRAIN_FILE_PATH, TEST_FILE_PATH)
    assert isinstance(loader.train, pd.DataFrame)
    assert isinstance(loader.test, pd.DataFrame)
    assert isinstance(loader.get_train_columns, list)
    assert loader.train.columns.values.tolist() == loader.test.columns.values.tolist()
    assert loader.train.columns.values.tolist() == FEATURES
    assert loader.train.dtypes.values.tolist() == loader.test.dtypes.values.tolist()
    assert loader.train.dtypes.values.tolist() == FEATURE_TYPES


@pytest.mark.parametrize("train_path, test_path, expected",
                         [('', TEST_FILE_PATH, "The train_path parameter provided is empty."),
                          (TRAIN_FILE_PATH, '', "The test_path parameter provided is empty.")])
def test_empty_file_paths(train_path: str, test_path: str, expected: str):
    """
    Tests the _validate_paths method and EmptyPath exception.
    :param train_path: File path for train data csv
    :param test_path: File path for test data csv
    :param expected: Expected message
    """
    try:
        Loader(train_path, test_path)
    except EmptyPath as e:
        assert e.message == expected


def test_trainer():
    """
    Tests the Trainer class.
    """
    loader = Loader(TRAIN_FILE_PATH, TEST_FILE_PATH)
    trainer = Trainer(loader)
    assert isinstance(trainer.categorical_transformer, TargetEncoder)
    assert isinstance(trainer.preprocessor, ColumnTransformer)
    assert isinstance(trainer.steps, list)
    assert isinstance(trainer.pipeline, Pipeline)
    assert trainer.pipeline.n_features_in_ == NUMBER_OF_FEATURES
    assert trainer.pipeline._estimator_type == MODEL_TYPE
    assert isinstance(trainer.pipeline._final_estimator, GradientBoostingRegressor)


# noinspection PyTypeChecker
def test_data_not_loaded():
    """
    Tests the DataNotLoaded exception.
    """
    try:
        Trainer(loader=None)
    except DataNotLoaded as e:
        assert e.message == "The data was not loaded. Please initialize the class Loader before training."


def test_evaluator():
    """
    Tests the Evaluator class.
    """
    loader = Loader(TRAIN_FILE_PATH, TEST_FILE_PATH)
    trainer = Trainer(loader)
    evaluator = Evaluator(loader, trainer)
    assert evaluator.predictions.dtype.name in TARGET_TYPE
    assert evaluator.target.dtype.name in TARGET_TYPE
    assert evaluator.predictions.size == evaluator.target.size
    assert evaluator.rmse < RMSE_THRESHOLD
    assert evaluator.mape < MAPE_THRESHOLD
    assert evaluator.mae < MAE_THRESHOLD


# noinspection PyTypeChecker
def test_model_not_trained():
    """
    Tests the ModelNotTrained exception.
    """
    try:
        Evaluator(loader=None, trainer=None)
    except ModelNotTrained as e:
        assert e.message == "The model was not trained. Please initialize the class Trainer before evaluating."
