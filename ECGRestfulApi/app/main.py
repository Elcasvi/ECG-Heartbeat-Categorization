from fastapi import FastAPI
from app.model.ECGInput import ECGInput
from app.model.services.ECGMultiClassClassifier import predict_ecg
from app.model.services.ECGTwoClassClassifier import predict_illness
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Configure CORS
orig_origins = [
    "http://localhost:3000",  # Replace with your React app URL
    "http://localhost",        # Add if you have more domains to allow
    "https://your-react-app-domain.com"  # Replace with your deployed app URL if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orig_origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

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

