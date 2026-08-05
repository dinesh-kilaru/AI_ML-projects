## Tools Used

This notebook utilized the following Python libraries:

*   **pandas**: For data manipulation and analysis.
*   **matplotlib.pyplot**: For creating static, interactive, and animated visualizations.
*   **seaborn**: For making statistical graphics based on matplotlib.
*   **sklearn.preprocessing.LabelEncoder**: For encoding categorical features.
*   **sklearn.preprocessing.StandardScaler**: For standardizing features by removing the mean and scaling to unit variance.
*   **sklearn.model_selection.train_test_split**: For splitting data into training and testing sets.
*   **sklearn.linear_model.LogisticRegression**: For implementing logistic regression.
*   **sklearn.neighbors.KNeighborsClassifier**: For implementing K-Nearest Neighbors classification.
*   **sklearn.tree.DecisionTreeClassifier**: For implementing Decision Tree classification.
*   **sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score)**: For evaluating model performance.
*   **joblib**: For saving and loading Python objects, particularly machine learning models.

## Process and Steps Taken

1.  **Data Loading**: The `Shopping Mall Customer Segmentation Data.csv` file was loaded into a pandas DataFrame.
2.  **Initial Data Inspection**: `df.head()`, `df.info()`, and `df.describe()` were used to get an initial understanding of the data's structure, types, and summary statistics.
3.  **Missing Values Handling**: Checked for missing values using `df.isnull().sum()` and filled numerical missing values with the mean using `df.fillna(df.mean(numeric_only=True), inplace=True)`. (Although no missing values were found, the code was included as a general preprocessing step).
4.  **Duplicate Handling**: Removed duplicate rows from the DataFrame using `df.drop_duplicates(inplace=True)`.
5.  **Categorical Feature Encoding**: The `Gender` column (an object type) was converted into numerical representation using `LabelEncoder`.
6.  **Exploratory Data Analysis (EDA)**:
    *   Histograms were plotted for all numerical features to visualize their distributions.
    *   A scatter plot was created to observe the relationship between `Age` and `Annual Income`.
    *   A bar plot of `Annual Income` value counts was generated.
7.  **Feature and Target Split**: The dataset was split into features (`X`) and the target variable (`y`), where `Spending Score` was designated as the target.
8.  **Data Splitting**: The data was divided into training and testing sets using `train_test_split` with a test size of 20% and a `random_state` of 42.
9.  **Feature Scaling**: `StandardScaler` was applied to `X_train` and `X_test` to standardize the features.
10. **Model Training**: Three different classification models were trained on the scaled training data:
    *   Logistic Regression (`LogisticRegression`)
    *   K-Nearest Neighbors (`KNeighborsClassifier`)
    *   Decision Tree Classifier (`DecisionTreeClassifier`)
11. **Model Prediction**: Predictions were made on the test set (`X_test`) using each trained model.
12. **Model Evaluation**: A custom `evaluate` function was used to calculate and print the Accuracy, Precision, Recall, and F1 Score for each model on the test set. Warnings were suppressed during evaluation.
13. **Best Model Saving**: The Decision Tree Classifier (which showed a slightly better F1 score than others in the trace output) was selected as the `bestmodel` and saved to a file named `bestmodel.pkl` using `joblib`.