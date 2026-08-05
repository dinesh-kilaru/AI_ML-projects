#### Wine Quality Prediction Project - Exploratory Data Analysis (EDA)

#### Overview



This notebook performs an initial Exploratory Data Analysis (EDA) on the Wine Quality dataset. The goal is to understand the dataset's structure, identify data quality issues, explore distributions of key features, and uncover relationships between variables before proceeding with any predictive modeling.



**Steps Performed**

1\. Data Loading

The dataset, WineQT.csv, was loaded into a pandas DataFrame.



2\. Data Inspection and Cleaning



Initial Overview: Performed checks using df.head(), df.tail(), df.info(), and df.describe() to understand the data's structure, data types, and basic statistics.



Missing Values: Confirmed that there were no missing values across any columns using df.isnull().sum().



Duplicate Rows: Identified and removed duplicate rows using df.drop\_duplicates(inplace=True) to ensure data uniqueness.

Outlier Handling: Outliers in the 'volatile acidity' column were identified using box plots and treated by capping them to their respective Interquartile Range (IQR) bounds to normalize the distribution.



3\. Exploratory Data Analysis (EDA)



Distribution Analysis: Visualized the distribution of various features:

Histograms were generated for 'fixed acidity' to observe its distribution.

Box plots were used to examine 'volatile acidity' before and after outlier capping.





Pie charts were created for 'citric acid' to show its value counts.

Aggregate Statistics: Computed mean, median, min, and max of 'fixed acidity' grouped by 'quality' to explore potential relationships.

Correlation Analysis: Generated and displayed a correlation matrix using df.corr() to understand the linear relationships between all numerical features. Key findings included correlations of 'fixed acidity', 'volatile acidity', and 'alcohol' with 'quality', as well as inter-correlations among other chemical properties.

Tools and Libraries Used





pandas: For data manipulation, loading, cleaning, and descriptive statistics.





numpy: For numerical operations, often used implicitly by pandas.

seaborn: For statistical data visualization.

matplotlib.pyplot: For creating static, interactive, and animated visualizations.



