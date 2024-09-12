from fastapi import FastAPI
from model.ECGInput import ECGInput
from model.services.ECGMultiClassClassifier import predict_ecg
from model.services.ECGTwoClassClassifier import predict_illness
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/predict_healthiness")
async def classify_healthiness(ecg_input: ECGInput):
    print("Inside function: classify_healthiness")
    ecg_data = ecg_input.data
    print("ecg_data: ",ecg_data)
    result = predict_illness(ecg_data)
    print("result: ",result)
    return result


@app.post("/predict_illness")
async def classify_illness (ecg_input:ECGInput):
    print("Inside function: classify_illness")
    ecg_data = ecg_input.data
    print("ecg_data: ",ecg_data)
    illness = predict_ecg(ecg_data)
    print("illness: ",illness)
    return illness

