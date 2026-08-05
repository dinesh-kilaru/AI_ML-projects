

### Student Data Analysis



This notebook performs an exploratory data analysis (EDA) on student performance data. The steps covered include data loading, initial inspection, data cleaning, and visualization.



#### Steps Performed



1. Data Loading\*\*: The `student\_data.csv` file was loaded into a pandas DataFrame



2\.    Displayed the first and last 5 rows of the DataFrame (`df.head()`, `df.tail()`).

&#x20;   \*   Checked the DataFrame information (`df.info()`) to understand data types and non-null counts.

&#x20;   \*   Generated descriptive statistics for numerical columns (`df.describe()`).

&#x20;   \*   Inspected column names (`df.columns`) and DataFrame shape (`df.shape`).





3\.  \*\*Data Cleaning\*\*: 

&#x20;   \*   Checked for and handled missing values (`df.isnull().sum()`, `df.dropna()`). No missing values were found.

&#x20;   \*   Checked for and removed duplicate rows (`df.duplicated().sum()`, `df.drop\_duplicates()`). No duplicate rows were found.

&#x20;   \*   Handled outliers in the 'absences' column using the IQR method, capping values at the lower and upper bounds.





4\.  \*\*Data Exploration and Visualization\*\*:

&#x20;   \*   Calculated and displayed the mean, median, min, and max ages.

&#x20;   \*   Grouped data by 'school' to analyze age statistics per school.

&#x20;   \*   Visualized the distribution of 'age' using a histogram (`df\["age"].hist()`).

&#x20;   \*   Created a scatter plot of 'Fedu' (Father's education) vs. 'age' (`df.plot.scatter()`).

&#x20;   \*   Visualized the distribution of students per 'school' using line and bar plots (`df\["school"].value\_counts().plot()`).

&#x20;   \*   Generated a correlation matrix for numerical features and visualized it as a heatmap (`sns.heatmap()`).



#### Tools Used



* pandas\*\*: For data loading, manipulation, and cleaning.
* &#x20;\*\*numpy\*\*: For numerical operations, especially during outlier treatment.
* &#x20; \*\*matplotlib.pyplot\*\*: For creating static, interactive, and animated visualizations.
* &#x20; \*\*seaborn\*\*: For making statistical graphics more attractive and informative.



```

