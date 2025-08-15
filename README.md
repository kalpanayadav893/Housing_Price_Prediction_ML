# Housing_Price_Prediction_ML
A machine learning project for housing price prediction using Linear, Lasso, and Ridge regression models, featuring data preprocessing, outlier handling, feature scaling, and performance evaluation.

# 🏠 Housing Price Prediction using Machine Learning

This project predicts **housing prices** using multiple regression models on the **Boston Housing Dataset**.  
It covers **data preprocessing, outlier handling, feature scaling, model training, evaluation, and comparison** of different regression techniques.

---

## 📌 Features of the Project
- **Data Preprocessing**
  - Removes duplicate entries.
  - Handles missing values with `SimpleImputer`.
  - Detects and removes outliers from `INDUS`.
  - Encodes categorical variables using `OneHotEncoder`.
  - Scales features using `MinMaxScaler`.

- **Exploratory Data Analysis**
  - Correlation heatmap to analyze feature relationships.
  - Boxplots for visualizing outliers.

- **Model Training**
  - Linear Regression
  - Lasso Regression
  - Ridge Regression

- **Model Evaluation**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
  - Residual plots for Linear Regression.
  - Feature importance chart for Linear Regression.
  - Model comparison bar plot.
