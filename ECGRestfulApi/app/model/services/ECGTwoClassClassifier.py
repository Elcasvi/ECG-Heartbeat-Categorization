import pickle
import numpy as np

# Load the illness classifier model
with open("app/MLModels/final_model_two_class_classifier.pkl", "rb") as f:
    illness_classifier  = pickle.load(f)

def predict_illness(data: list[float]) -> str:
    # Convert input to NumPy array
    data_array = np.array(data).reshape(1, -1)

    # Perform prediction
    prediction = illness_classifier.predict(data_array)

    # Map prediction to illness type (adjust based on your model)
    if prediction==0.0:
        print("In case 0")
        return "healthy"
    elif prediction==1.0:
        print("In case 1")
        return "sick"