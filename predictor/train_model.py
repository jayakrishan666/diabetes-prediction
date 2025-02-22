import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("predictor/diabetes.csv")

# Select features and target variable
X = df[['BMI', 'Age']]
y = df['Outcome']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'predictor/diabetes_model.pkl')

print("Model training completed and saved as diabetes_model.pkl")
