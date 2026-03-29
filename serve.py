from fastapi import FastAPI
from pydantic import BaseModel
from clearml import Model
import pickle

app = FastAPI()

# Загружаем модель из ClearML Registry
print("Загружаем модель из ClearML...")
model_obj = Model(model_id="40828760798b4198b0c09e6407af06a4")
path = model_obj.get_local_copy()

with open(path, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
vectorizer = bundle["vectorizer"]
print("Модель загружена!")

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Request):
    vec = vectorizer.transform([req.text])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return {"label": label, "confidence": round(float(proba), 3)}

@app.get("/health")
def health():
    return {"status": "ok"}