import pickle
import uvicorn as uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import Loader, Trainer, Evaluator
import logging.handlers
import datetime

MODEL_FILE_NAME = "trained_model.pkl"


class FilePaths(BaseModel):
    train_path: str
    test_path: str


class ModelPath(BaseModel):
    model_path: str


app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.handlers.RotatingFileHandler("app.log", maxBytes=1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@app.post("/train")
def train_model(file_paths: FilePaths):
    try:
        try:
            start_time = datetime.datetime.now()
            loader = Loader(file_paths.train_path, file_paths.test_path)
            logger.info(f"Data loading completed. Elapsed time: {datetime.datetime.now() - start_time}s")
        except Exception as e:
            logger.error("Error occurred during data loading: %s", str(e))
            return

        try:
            start_time = datetime.datetime.now()
            trainer = Trainer(loader)
            logger.info(f"Model training completed. Elapsed time: {datetime.datetime.now() - start_time}s")
        except Exception as e:
            logger.error("Error occurred during model training: %s", str(e))
            return

        evaluator = Evaluator(loader, trainer)
        logger.info(f"Training metrics: RMSE: {evaluator.rmse} - MAPE: {evaluator.mape} - MAE: {evaluator.mae} ")

        model_path = MODEL_FILE_NAME
        with open(model_path, "wb") as file:
            pickle.dump(trainer.pipeline, file)

        return ModelPath(model_path=model_path)

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
