import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/boston_housing.csv")
df= pd.read_csv(file_path)

# Display basic info
print(df.info())  
print(df.head())
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values[missing_values > 0])

# Impute missing numerical values with the median
num_imputer = SimpleImputer(strategy="median")
df.iloc[:, :] = num_imputer.fit_transform(df)

# Visualize outliers using boxplot
plt.figure(figsize=(12, 6))
df.boxplot(rot=45)
plt.title("Boxplot for Outlier Detection")
plt.show()
# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal on selected features
outlier_columns = ["rm", "lstat", "crim","zn", "dis", "ptratio", "b", "medv"]
for col in outlier_columns:
    df = remove_outliers(df, col)
    # Apply one-hot encoding to categorical columns

df['chas'] = df['chas'].astype('category')
print(df.dtypes)

df = pd.get_dummies(df, columns = ['chas'], prefix='chas', drop_first = True)

display(df.head)
#Standardization (Recommended for Linear Regression)
scaler = StandardScaler()
# Define features (X) and target variable (y)
X = df.drop('medv', axis=1)
y = df['medv']
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
display(X_scaled)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Save processed data
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)
