#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

###Load the Preprocessed Dataset
def read_file(filename):
    filepath = '../Data/'+str(filename)
    return pd.read_csv(filepath)

X_test = read_file('X_test.csv')
y_test = read_file('y_test.csv')
X_train = read_file('X_train.csv')
y_train = read_file('y_train.csv')

print(len(X_test), len(X_train), len(y_test), len(y_train))
# Initialize the model
model = LinearRegression()

# Train (fit) the model
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Print first five predictions
print("Predicted Prices:", y_pred[:5])
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²) Score
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

#Hyperparameter Tuning (Random Forest Rgressor)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [None, 10, 20, 30],  # Maximum depth of trees
    "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4]  # Minimum samples at leaf node
}

# Initialize GridSearchCV
rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="neg_mean_squared_error",
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# Fit GridSearchCV
rf_grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", rf_grid_search.best_params_)
print("Best Score:", np.sqrt(-rf_grid_search.best_score_))  # Convert to RMSE

# Evaluate on test set
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Test Set RMSE:", rmse)

#Hyperparameter Tuning (Ridge Rigresson)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Define the Ridge Regression model
ridge = Ridge()

# Define hyperparameter grid for tuning (alpha values)
param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Using GridSearchCV for exhaustive search
ridge_grid_search = GridSearchCV(ridge, param_grid, scoring="neg_mean_squared_error", cv=5)
ridge_grid_search.fit(X_train, y_train)

# Best parameters and model
best_ridge = ridge_grid_search.best_estimator_
best_alpha = ridge_grid_search.best_params_["alpha"]

# Evaluate the best model
y_pred_ridge = best_ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Print Results
print(f"Best Alpha: {best_alpha}")
print(f"MSE (Ridge): {mse_ridge}")
print(f"R² Score (Ridge): {r2_ridge}")


#Hyperparameter Tuning(Gradient Boosting Regressor)
models = {
    'Ridge': ridge_grid_search,
    'RandomForest': rf_grid_search,
    'GradientBoost': gb_grid_search
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MSE': mean_squared_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Best Params': model.best_params_
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('MSE'))

import joblib 
# Save the best performing model
joblib.dump(rf_grid_search, "../data/best_model.pkl")
print("\nSaved best model to ../data/best_model.pkl")