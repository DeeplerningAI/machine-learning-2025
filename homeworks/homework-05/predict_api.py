from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# 1. Load model pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# 2. Define data model
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 3. Initialize FastAPI app
app = FastAPI(title="Lead Conversion Model")

@app.get("/")
def root():
    return {"message": "API is running"}

# 4. POST endpoint for prediction
@app.post("/predict")
def predict(lead: Lead):
    client_dict = {
        "lead_source": lead.lead_source,
        "number_of_courses_viewed": lead.number_of_courses_viewed,
        "annual_income": lead.annual_income,
    }

    X = [client_dict]
    proba = model.predict_proba(X)[0, 1]
    return {"conversion_probability": round(float(proba), 3)}

# 5. Run manually with: uvicorn predict_api:app --reload --port 9696
