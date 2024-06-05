# Cryptocurrency Data Processing and Model Training Notebook

## Overview

This notebook demonstrates the process of preprocessing cryptocurrency data, followed by training and evaluating various machine learning models for classification and regression tasks. The primary focus is on transforming and standardizing the data, handling categorical variables through one-hot encoding, and using different models to predict the cryptocurrency type and price.

## Dataset

The dataset contains historical cryptocurrency data with the following columns:
- `cryptoName`: Name of the cryptocurrency (e.g., BTC, ETH).
- `date`: Date of the data point.
- `open`: Opening price.
- `close`: Closing price.
- `volume`: Trading volume.
- `average`: Average price.
- `change`: Price change.
- `openClose`: Difference between open and close prices.
- `dailyReturn`: Daily return.
- `dayOfWeek`: Day of the week.
- `lagBy1Day`: Lagged price by 1 day.
- `lagByWeek`: Lagged price by 1 week.

## Data Preprocessing

### One-Hot Encoding
Categorical columns, specifically `cryptoName`, are transformed into numerical format using one-hot encoding.

### Data Type Transformation
The `date` column is converted from `datetime64` to `int64` for compatibility with machine learning algorithms.

### Data Standardization
Numerical columns are standardized to have zero mean and unit variance.

### Handling Missing Values
Missing values in numerical columns are imputed using the median value.

## Data Visualization

A time series plot of trading volume against date is created for each cryptocurrency to visualize trends and patterns in the data.

## Model Training and Evaluation

### Feature and Target Preparation
- For classification tasks, `cryptoName` is used as the target variable.
- For regression tasks, `average` price is used as the target variable.

### Machine Learning Models
Several machine learning models are trained and evaluated using cross-validation:
1. **Linear Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **Support Vector Machine (SVM) with RBF Kernel**
5. **K-Nearest Neighbors (KNN)**

### Model Performance
- **Linear SVM:** Achieved an accuracy of 56.88%.
- **Random Forest:** Achieved an accuracy of 99.27%.
- **XGBoost:** Achieved an accuracy of 99.51%.
- **SVM with RBF Kernel:** Achieved an accuracy of 52.33%.
- **KNN:** Achieved an accuracy of 48.14%.

## Usage

To use this notebook, follow these steps:
1. Ensure you have the required libraries installed (`pandas`, `numpy`, `sklearn`, `seaborn`, `matplotlib`, `xgboost`).
2. Load your dataset into a pandas DataFrame.
3. Follow the steps outlined for data preprocessing, model training, and evaluation.
4. Adjust the model parameters and preprocessing steps as needed for your specific dataset and use case.

## Conclusion

This notebook provides a comprehensive guide for preprocessing cryptocurrency data and training various machine learning models for classification and regression tasks. The detailed steps and code examples can be adapted to other datasets and machine learning problems.

## Future Work

Potential improvements and future directions include:
- Hyperparameter tuning for each model to optimize performance.
- Incorporating additional features and external data sources to enhance model accuracy.
- Experimenting with deep learning models for time series forecasting and classification.
