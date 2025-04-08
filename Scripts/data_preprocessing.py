import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(df.info())
    print(df.head())
    return df

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing values per column:\n", missing_values[missing_values > 0])
    return missing_values

def impute_missing_values(df):
    print("Imputing missing values...")
    num_imputer = SimpleImputer(strategy="median")
    df.iloc[:, :] = num_imputer.fit_transform(df)
    return df

def visualize_outliers(df):
    print("Visualizing outliers...")
    plt.figure(figsize=(12, 6))
    df.boxplot(rot=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def apply_outlier_removal(df, columns):
    for col in columns:
        df = remove_outliers(df, col)
    return df

def encode_categorical(df):
    print("Encoding categorical variables...")
    df['chas'] = df['chas'].astype('category')
    df = pd.get_dummies(df, columns=['chas'], prefix='chas', drop_first=True)
    return df

def standardize_features(X):
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled

def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print("Processed data saved.")

def main():
    file_path = "../data/boston_housing.csv"
    output_dir = "../data"

    df = load_data(file_path)
    check_missing_values(df)
    df = impute_missing_values(df)
    visualize_outliers(df)

    outlier_columns = ["rm", "lstat", "crim", "zn", "dis", "ptratio", "b", "medv"]
    df = apply_outlier_removal(df, outlier_columns)

    df = encode_categorical(df)

    X = df.drop('medv', axis=1)
    y = df['medv']
    X_scaled = standardize_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)

if __name__ == "__main__":
    main()
