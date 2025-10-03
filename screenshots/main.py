from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from ml.data import process_data
from ml.model import inference, load_model

app = FastAPI(title="D501 Inference API")

model, encoder, lb = load_model()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class CensusRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

@app.get("/")
def root():
    return {"message": "Welcome to the salary prediction API. POST to /predict"}

@app.post("/predict")
def predict(req: CensusRequest):
    row = req.model_dump(by_alias=True)
    df = pd.DataFrame([row])
    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    yhat = inference(model, X)
    try:
        label = lb.inverse_transform(yhat)[0]
    except Exception:
        label = int(yhat[0])
    return {"prediction": int(yhat[0]), "label": str(label)}
