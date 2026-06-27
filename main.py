from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("random_forest_classifier.pkl")
scaler = joblib.load("standard_scaler.pkl")


class BankInput(BaseModel):
    age:int
    job:str
    marital:str
    education:str
    default:str
    balance:int
    housing:str
    loan:str
    contact:str
    day:int
    month:str
    duration:int
    campaign:int
    pdays:int
    previous:int
    poutcome:str


job_map={
    "management":0,
    "technician":1,
    "entrepreneur":2,
    "blue-collar":3,
    "unknown":4,
    "retired":5,
    "admin.":6,
    "services":7,
    "self-employed":8,
    "unemployed":9,
    "housemaid":10,
    "student":11
}

marital_map={
    "married":0,
    "single":1,
    "divorced":2
}

education_map={
    "tertiary":0,
    "secondary":1,
    "unknown":2,
    "primary":3
}

binary_default={
    "no":0,
    "yes":1
}

housing_map={
    "yes":0,
    "no":1
}

loan_map={
    "no":0,
    "yes":1
}

contact_map={
    "unknown":0,
    "cellular":1,
    "telephone":2
}

poutcome_map={
    "unknown":0,
    "failure":1,
    "other":2,
    "success":3
}

month_map={
"jan":1,
"feb":2,
"mar":3,
"apr":4,
"may":5,
"jun":6,
"jul":7,
"aug":8,
"sep":9,
"oct":10,
"nov":11,
"dec":12
}


@app.get("/")
def home():
    return {"message":"Bank Marketing Prediction API Running"}


@app.post("/predict")
def predict(data:BankInput):

    row={
        "age":data.age,
        "job":job_map[data.job.lower()],
        "marital":marital_map[data.marital.lower()],
        "education":education_map[data.education.lower()],
        "default":binary_default[data.default.lower()],
        "balance":data.balance,
        "housing":housing_map[data.housing.lower()],
        "loan":loan_map[data.loan.lower()],
        "contact":contact_map[data.contact.lower()],
        "day":data.day,
        "month":month_map[data.month.lower()],
        "duration":data.duration,
        "campaign":data.campaign,
        "pdays":data.pdays,
        "previous":data.previous,
        "poutcome":poutcome_map[data.poutcome.lower()]
    }

    df=pd.DataFrame([row])

    transformed=scaler.transform(df)

    prediction=model.predict(transformed)[0]

    confidence=max(model.predict_proba(transformed)[0])

    result="Will Subscribe" if prediction==1 else "Will Not Subscribe"

    return {
        "prediction":result,
        "confidence":round(float(confidence),4)
    }


import uvicorn
import os

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8000))
    uvicorn.run(app,host="0.0.0.0",port=port)