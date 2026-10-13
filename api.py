from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Mental Health Scan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the trained model
trained_model = joblib.load("risk_model.joblib")

# 1. Schema for 14 depression questions
class PoolAnswers(BaseModel):
    Q3A: int
    Q5A: int
    Q10A: int
    Q13A: int
    Q16A: int
    Q17A: int
    Q21A: int
    Q24A: int
    Q26A: int
    Q31A: int
    Q34A: int
    Q37A: int
    Q38A: int
    Q42A: int


@app.post("/api/calculate_risk")
def calculate_risk(answers: PoolAnswers):
    # Convert JSON payload to a DataFrame
    data_dict = answers.dict()
    user_data = pd.DataFrame([data_dict])

    # 3. The scoring system is 0-3 instead of 1-4
    user_data = user_data - 1

    # 4. We have five risk levels, so the model returns five probability values
    probabilities = trained_model.predict_proba(user_data)[0]

    # 5. Convert results to percentages for the frontend
    return {
        "status": "success",
        "analysis_results": {
            "normal_probability": round(probabilities[0] * 100, 2),
            "mild_risk_probability": round(probabilities[1] * 100, 2),
            "moderate_risk_probability": round(probabilities[2] * 100, 2),
            "severe_risk_probability": round(probabilities[3] * 100, 2),
            "very_severe_risk_probability": round(probabilities[4] * 100, 2),
        },
    }
