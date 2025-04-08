import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

def read_file(filename):
    filepath = os.path.join('../Data', filename)
    return pd.read_csv(filepath)

def load_datasets():
    X_test = read_file('X_test.csv')
    y_test = read_file('y_test.csv')
    X_train = read_file('X_train.csv')
    y_train = read_file('y_train.csv')
    
    print(f"Dataset sizes - X_train: {len(X_train)}, X_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")
    return X_train, X_test, y_train, y_test

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    
    return y_pred, mse, r2

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    # Histogram of residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution")
    plt.show()

    # Residuals vs Predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color="blue", alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()

def compare_selected_features(X_train, X_test, y_train, y_test):
    selected_features = ['rm', 'lstat', 'ptratio']
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    model_sel = LinearRegression()
    model_sel.fit(X_train_sel, y_train)
    
    y_pred_sel = model_sel.predict(X_test_sel)
    mse_sel = mean_squared_error(y_test, y_pred_sel)
    r2_sel = r2_score(y_test, y_pred_sel)

    print(f"Selected Features - MSE: {mse_sel:.2f}, R²: {r2_sel:.2f}")

def main():
    X_train, X_test, y_train, y_test = load_datasets()

    print("\nTraining and evaluating full model...")
    model = train_model(X_train, y_train)
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)

    print("\nPlotting residuals...")
    plot_residuals(y_test, y_pred)

    print("\nComparing with selected features...")
    compare_selected_features(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
