from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ModelInput(BaseModel):
    user: int
    card: int
    # Các trường khác nếu muốn


@app.post("/predict")
def predict(data: ModelInput):
    # Fake logic
    if data.card % 2 == 0:
        pred = "fraud"
    else:
        pred = "not_fraud"
    return {"prediction": pred}


# Chạy: uvicorn fastapi_model:app --host 0.0.0.0 --port 8000
