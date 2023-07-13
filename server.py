import pandas as pd
import numpy as np
import uvicorn as uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from model import Loader, Trainer, Evaluator
import logging.handlers
import datetime
import os
from dotenv import load_dotenv
import pickle


class FilePaths(BaseModel):
    train_path: str
    test_path: str


class ModelPath(BaseModel):
    model_path: str


class PropertyData(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: int
    n_bathroom: int
    latitude: float
    longitude: float


load_dotenv()
MODEL_FILE_NAME = "trained_model.pkl"
API_KEY = os.getenv('API_KEY')
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.handlers.RotatingFileHandler("app.log", maxBytes=1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )


@app.post("/train")
def train_model(file_paths: FilePaths, api_key: str = Depends(validate_api_key)):
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


@app.post("/predict")
def predict_price(property_data: PropertyData, api_key: str = Depends(validate_api_key)):
    with open("trained_model.pkl", "rb") as file:
        trained_model = pickle.load(file)

    try:
        features = pd.DataFrame(
            data={key: [value] for key, value in property_data.dict().items()},
            columns=property_data.__annotations__.keys()
        )

        predicted_price = trained_model.predict(features)[0]

        logger.info(f"Prediction finished successfully. The predicted value was {predicted_price}")
        return {"predicted_price": predicted_price}

    except Exception as e:
        logger.error("Error occurred during price prediction: %s", str(e))
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
