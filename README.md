# Road-accidents-model-DataSci
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load the dataset
file_path = "/content/drive/MyDrive/Colab CSV Files/Kenya Road Accidents Dataset.csv"
df = pd.read_csv(file_path, delimiter="\t")

# Define dependent and independent variables
X = df.drop(columns=["Accident_ID", "Accident_Severity_(1_10)"])  # Features
y = df["Accident_Severity_(1_10)"]  # Target

# Identify categorical and numerical features
categorical_features = ["Weather", "Road_Type", "Time_of_Day", "Vehicle_Type", "Road_Condition", "Location"]
numerical_features = ["Speed_Limit_(km/h)", "Driver_Age"]

# Preprocessing: One-hot encoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # Handle unknown categories
    ]
)

# Create a pipeline with preprocessing and regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model for future use
joblib.dump(model, "road_accident_severity_model.pkl")

# Define a hypothetical accident scenario for prediction
hypothetical_data = pd.DataFrame({
    "Weather": ["Rainy"],
    "Road_Type": ["Highway"],
    "Speed_Limit_(km/h)": [90],
    "Driver_Age": [30],
    "Time_of_Day": ["Morning"],
    "Vehicle_Type": ["Car"],
    "Road_Condition": ["Paved"],
    "Location": ["Nairobi"]
})

# Ensure hypothetical data goes through the same transformation pipeline
predicted_severity = model.predict(hypothetical_data)

# Display the prediction result
print(f"Predicted Accident Severity: {predicted_severity[0]:.2f}")
