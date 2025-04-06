import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

###Load the Preprocessed Dataset
def read_file(filename):
    filepath = '../Data/'+str(filename)
    return pd.read_csv(filepath)

X_test = read_file('X_test.csv')
y_test = read_file('y_test.csv')
X_train = read_file('X_train.csv')
y_train = read_file('y_train.csv')

print(len(X_test), len(X_train), len(y_test), len(y_train))

print("X_test data: \n", X_test)
print("y_test data: \n", y_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

##Evaluate Model
# Predict on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color="blue")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.show()

# Residuals vs Predictions plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color="blue", alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

#Compare Performance with Different Feature Sets
# Try with fewer features
selected_features = ['rm', 'lstat', 'ptratio']  # Select important features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train new model
model_selected = LinearRegression()
model_selected.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred_selected = model_selected.predict(X_test_selected)
mse_selected = mean_squared_error(y_test, y_pred_selected)
r2_selected = r2_score(y_test, y_pred_selected)

print(f"Selected Features - MSE: {mse_selected:.2f}, R²: {r2_segitlected:.2f}")