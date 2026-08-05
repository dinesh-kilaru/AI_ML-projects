### Titanic Survival Prediction - Exploratory Data Analysis (EDA)

### Project Overview



This notebook performs an Exploratory Data Analysis (EDA) on the tested.csv dataset, which is part of the Titanic: Machine Learning from Disaster Kaggle competition. The goal of this EDA is to understand the dataset's structure, identify patterns, and prepare the data for further machine learning model development.

#### 

#### Dataset

The dataset used is tested.csv, containing information about passengers on the Titanic. It includes various features such as PassengerId, Survival status (note: 'Survived' column in tested.csv is usually for submission and may not reflect actual survival for the test set), Pclass, Name, Sex, Age, SibSp (siblings/spouses aboard), Parch (parents/children aboard), Ticket, Fare, Cabin, and Embarked (port of embarkation).

#### 

#### Steps Performed

1\. Data Loading

The dataset was loaded into a pandas DataFrame from the /content/tested.csv file.



2\. Initial Data Inspection

Displayed the first and last 5 rows (.head(), .tail()) to get a quick overview.

Checked the data types and non-null counts (.info()) to identify missing values and data types.

Examined the dataset dimensions (.shape).

Listed column names (.columns).

Generated descriptive statistics for numerical columns (.describe()).



3\. Missing Values and Duplicates

Identified the count of missing values per column (.isnull().sum()). Significant missing values were found in 'Age', 'Fare', and 'Cabin'.

Checked for duplicate rows in the dataset, confirming there were none.



4\. Data Cleaning and Preparation

Missing Value Imputation: 'Age' was imputed with the median, and 'Fare' was imputed with its median.

Outlier Capping: Outliers in 'Age' and 'Fare' were capped using the Interquartile Range (IQR) method to handle extreme values.



5\. Exploratory Data Analysis (EDA)

Univariate Analysis:

Histograms were plotted for 'Pclass' and 'Age' to visualize their distributions.

A bar plot was generated for 'Sex' to show the distribution of genders.

A boxplot for 'PassengerId' (though not directly informative for this feature, it was part of the exploration).

Bivariate Analysis:

A scatter plot of 'Age' vs 'Fare' was created to explore their relationship.

#### 

#### Survival Rate Analysis:

Calculated the overall survival rate.

Analyzed survival rates by 'Sex'.

Analyzed survival rates by 'Pclass'.

Analyzed survival rates by the combination of 'Sex' and 'Pclass'.

#### 

#### Key Findings

The dataset has 418 entries and 12 columns.

'Age' and 'Fare' have missing values, and 'Cabin' has a high percentage of missing values.

The 'Age' distribution is skewed, with a majority of passengers being young adults. The 'Fare' distribution is also skewed with many low fares.

'Pclass' 3 has the most passengers.

The distribution of 'Sex' shows more males than females.

Important Note: In this specific tested.csv dataset, all females are shown as 'Survived=1' and all males as 'Survived=0'. This is a common characteristic of test datasets in Kaggle competitions where the 'Survived' column is a placeholder, and actual predictions are to be submitted. This indicates that direct survival analysis on this specific column in tested.csv might not reflect the real-world survival probabilities typically seen in the training data.



#### Tools and Libraries Used

pandas: For data loading, manipulation, and analysis.

numpy: For numerical operations.

matplotlib.pyplot: For creating static, interactive, and animated visualizations.

seaborn: For making statistical graphics based on matplotlib.

