import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Rain Prediction Project")


model = joblib.load("artifacts/model.pkl")


encoders = {}
encoder_cols = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

for col in encoder_cols:
    encoders[col] = joblib.load(f"artifacts/encoders/{col}.pkl")

encoders["RainTomorrow"] = joblib.load(
    "artifacts/encoders/RainTomorrow.pkl"
)


class WeatherInput(BaseModel):
    Date: str
    Location: str
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float | None = None
    Sunshine: float | None = None
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float | None = None
    Cloud3pm: float | None = None
    Temp9am: float
    Temp3pm: float
    RainToday: str


@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/predict")
def predict(data: WeatherInput):
    try:
        df = pd.DataFrame([data.dict()])


        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df.drop("Date", axis=1, inplace=True)

        for col in encoder_cols:
            df[col] = encoders[col].transform(df[col].astype(str))

    
        df = df.fillna(0)


        pred = model.predict(df)

        result = encoders["RainTomorrow"].inverse_transform(pred)

        return {"RainTomorrow": result[0]}

    except:
        raise HTTPException(status_code=500, detail="Error during prediction")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
