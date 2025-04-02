import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"D:\Assignment\Machine Lerning\machine-learning-introduction-Sabuna-Gamal\Data\BostonHousing_preprocessed.csv"
df = pd.read_csv(file_path)
# Load the preprocessed dataset
X_test = pd.read_csv(r"D:\Assignment\Machine Lerning\machine-learning-introduction-Sabuna-Gamal\Data\X_test.csv")
X_train = pd.read_csv(r"D:\Assignment\Machine Lerning\machine-learning-introduction-Sabuna-Gamal\Data\X_train.csv")
y_test = pd.read_csv(r"D:\Assignment\Machine Lerning\machine-learning-introduction-Sabuna-Gamal\Data\y_test.csv")
y_train = pd.read_csv(r"D:\Assignment\Machine Lerning\machine-learning-introduction-Sabuna-Gamal\Data\y_train.csv")

# Display the first few rows
print(X_test.head())
print(X_train.head())
print(y_test.head())
print(y_train.head())

# Define features (X) and target variable (y)
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Save the trained model
joblib.dump(model, "linear_regression_model.pkl")
