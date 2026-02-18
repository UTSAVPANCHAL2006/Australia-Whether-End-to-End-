import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ------------------ GLOBALS ------------------
model = None
encoders = {}

encoder_cols = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

# ------------------ LIFESPAN (startup) ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders

    # Load model
    model = joblib.load(
        os.path.join(ARTIFACTS_DIR, "model", "model.pkl")
    )

    # Load encoders
    for col in encoder_cols:
        encoders[col] = joblib.load(
            os.path.join(ARTIFACTS_DIR, "encoders", f"{col}.pkl")
        )

    encoders["RainTomorrow"] = joblib.load(
        os.path.join(ARTIFACTS_DIR, "encoders", "RainTomorrow.pkl")
    )

    print("✅ Model and encoders loaded successfully")
    yield
    print("🛑 App shutdown")

# ------------------ FASTAPI APP ------------------
app = FastAPI(
    title="Rain Prediction Project",
    lifespan=lifespan
)

# ------------------ SCHEMA ------------------
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

# ------------------ ROUTES ------------------
@app.get("/")
def home():
    return {"message": "Rain Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ ENTRYPOINT ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
