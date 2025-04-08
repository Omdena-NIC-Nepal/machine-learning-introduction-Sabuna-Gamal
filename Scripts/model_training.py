# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the preprocessed dataset
def read_file(filename):
    filepath = '../Data/' + str(filename)
    return pd.read_csv(filepath)

# Load datasets
X_test = read_file('X_test.csv')
y_test = read_file('y_test.csv')
X_train = read_file('X_train.csv')
y_train = read_file('y_train.csv')

print(f"Data Sizes -> X_train: {len(X_train)}, X_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")

### Linear Regression ###
print("\n--- Linear Regression ---")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

y_pred_linear = linear_model.predict(X_test)
print("First 5 Predictions:", y_pred_linear[:5])

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression - MSE:", mse_linear)
print("Linear Regression - R²:", r2_linear)

### Hyperparameter Tuning: Random Forest ###
print("\n--- Random Forest Tuning ---")
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Best RF Params:", rf_grid_search.best_params_)
print("RF CV RMSE:", np.sqrt(-rf_grid_search.best_score_))
print("RF Test RMSE:", rmse_rf)

### Hyperparameter Tuning: Ridge Regression ###
print("\n--- Ridge Regression Tuning ---")
ridge = Ridge()
ridge_param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, scoring="neg_mean_squared_error", cv=5)
ridge_grid_search.fit(X_train, y_train)

best_ridge = ridge_grid_search.best_estimator_
best_alpha = ridge_grid_search.best_params_["alpha"]
y_pred_ridge = best_ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Best Alpha (Ridge):", best_alpha)
print("Ridge - MSE:", mse_ridge)
print("Ridge - R²:", r2_ridge)

### Hyperparameter Tuning: Gradient Boosting ###
print("\n--- Gradient Boosting Tuning ---")
gb_model = GradientBoostingRegressor(random_state=42)
gb_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

gb_grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=gb_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)
gb_grid_search.fit(X_train, y_train)

best_gb_model = gb_grid_search.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("Best GB Params:", gb_grid_search.best_params_)
print("GB - MSE:", mse_gb)
print("GB - R²:", r2_gb)

### Compare Models ###
print("\n--- Model Comparison ---")
models = {
    'LinearRegression': linear_model,
    'RandomForest': rf_grid_search,
    'Ridge': ridge_grid_search,
    'GradientBoosting': gb_grid_search
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MSE': mean_squared_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Best Params': getattr(model, 'best_params_', 'N/A')
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('MSE'))

# Save best model (example: Random Forest)
joblib.dump(best_rf_model, "../data/best_model.pkl")
print("\nSaved best model to ../data/best_model.pkl")
