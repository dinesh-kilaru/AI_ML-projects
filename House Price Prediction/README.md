# Housing Price Prediction Project: README

This notebook details a comprehensive analysis and modeling effort to predict housing prices using the `Housing.csv` dataset.

## Tools and Libraries Used
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
- `seaborn`: For drawing attractive statistical graphics.
- `sklearn.model_selection`: For splitting data into training and testing sets.
- `sklearn.linear_model.LinearRegression`: For implementing Linear Regression models.
- `sklearn.metrics`: For evaluating model performance (MAE, MSE, RMSE, R-squared).
- `joblib`: For saving and loading trained models.

## Process Overview
1.  **Data Loading**: The `Housing.csv` dataset was loaded into a pandas DataFrame.
2.  **Exploratory Data Analysis (EDA)**: Initial insights into the dataset's structure, statistics, and distributions were gathered.
3.  **Data Preprocessing**: Categorical features were converted into numerical format using one-hot encoding.
4.  **Model Training**: Both Simple Linear Regression (using only 'area' as a feature) and Multiple Linear Regression (using all relevant features) models were trained.
5.  **Model Evaluation & Comparison**: The models were evaluated using MAE, MSE, RMSE, and R-squared metrics. The Multiple Linear Regression model was found to significantly outperform the Simple Linear Regression model.
6.  **Prediction**: Predictions were made on a subset of unseen test data using the better-performing Multiple Linear Regression model.
7.  **Model Saving**: The best-performing Multiple Linear Regression model was saved using `joblib` for future use.

