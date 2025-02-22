import numpy as np
import joblib
from django.shortcuts import render
from django.http import JsonResponse

# Load trained model
model = joblib.load("predictor/diabetes_model.pkl")

def predict(request):
    if request.method == "POST":
        try:
            # Get user inputs
            bmi = float(request.POST.get("BMI"))
            age = int(request.POST.get("Age"))

            # Prepare input data
            input_data = np.array([[bmi, age]])

            # Predict diabetes risk
            prediction = model.predict(input_data)[0]

            # Convert prediction to meaningful output
            if prediction < 0.5:
                result_text = "Low risk of diabetes."
            else:
                result_text = "High risk of diabetes."

            return JsonResponse({"prediction": result_text})

        except Exception as e:
            return JsonResponse({"error": str(e)})

def index(request):
    return render(request, 'predictor/index.html')
