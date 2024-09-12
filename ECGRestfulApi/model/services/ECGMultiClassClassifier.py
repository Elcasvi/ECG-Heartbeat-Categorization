import pickle
from typing import Any


import numpy as np


# Load the ECG classifier model
with open("MLModels/final_model_multiclass_classifier.pkl", "rb") as f:
    ecg_classifier = pickle.load(f)

def predict_ecg(data: list[float]) -> str:
    # Convert input to NumPy array
    data_array = np.array(data).reshape(1, -1)

    # Perform prediction
    prediction = ecg_classifier.predict(data_array)
    classification=''
    if prediction == 1.0:
        classification = 'S'
    elif prediction == 2.0:
        classification = 'Q'
    elif prediction == 3.0:
        classification = 'V'
    elif prediction == 4.0:
        classification = 'F'
    else:
        classification = 'Not Defined'

    return classification
    # Return result

