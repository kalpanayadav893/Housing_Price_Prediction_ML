# Import Required Libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Load Dataset
df = pd.read_csv(r"C:\Users\kalpa\Downloads\HousingData.csv")
print("Original shape:", df.shape)
print(df.head(3))

# Data Cleaning
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

print("\n🕳 Missing values:\n", df.isnull().sum())
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Summary Statistics
print("\nSummary statistics:\n", df.describe())

# Boxplot and Outlier Removal (INDUS)
if 'INDUS' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y='INDUS', data=df)
    plt.title("Boxplot of INDUS")
    plt.show()

    Q1 = df['INDUS'].quantile(0.25)
    Q3 = df['INDUS'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['INDUS'] > lower_bound) & (df['INDUS'] < upper_bound)]
    print("After removing INDUS outliers:", df.shape)

# One-Hot Encoding (CHAS)
if 'CHAS' in df.columns:
    df['CHAS'] = df['CHAS'].astype('object')

cat_cols = df.select_dtypes(include='object').columns.tolist()

if cat_cols:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df.drop(columns=cat_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

# Feature & Target
if 'MEDV' not in df.columns:
    raise ValueError("'MEDV' (target) column not found!")

X = df.drop(columns='MEDV')
y = df['MEDV']

# Correlation Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train/Test Split:", X_train.shape, X_test.shape)

# Model Training & Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=50, max_iter=1000, tol=0.1),
    "Ridge Regression": Ridge(alpha=50, max_iter=1000, tol=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R² Score": round(r2, 2)
    })

    if name == "Linear Regression":
        # Plot Residuals
        plt.figure(figsize=(6, 4))
        sns.residplot(x=preds, y=y_test - preds, lowess=True, color="g")
        plt.title("Residual Plot - Linear Regression")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.show()

        # Plot Feature Importance
        coef = pd.Series(model.coef_, index=X.columns).sort_values()
        plt.figure(figsize=(8, 6))
        coef.plot(kind='barh')
        plt.title("Feature Importance (Linear Regression Coefficients)")
        plt.show()

# Results
results_df = pd.DataFrame(results)
print("\nModel Comparison:\n")
print(results_df)

# Model Comparison Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='MAE', palette='Set2')
plt.title("Model Comparison - MAE")
plt.ylabel("Mean Absolute Error")
plt.show()
