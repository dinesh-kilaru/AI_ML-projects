### &#x20;Diamond Price Analysis

#### Overview

This notebook performs an exploratory data analysis (EDA) on a dataset of diamonds to understand their characteristics and the factors influencing their prices. The goal is to identify patterns, relationships, and prepare the data for further analysis or model building.



#### Steps Taken

1\. Data Loading

The diamonds.csv dataset was loaded into a pandas DataFrame using pd.read\_csv().



2\. Initial Data Exploration



df.head(): Displayed the first few rows to get a glimpse of the data structure.



df.info(): Checked data types, non-null counts, and memory usage. It was observed that there were no missing values initially and columns were of appropriate types (float, int, object).



df.describe(): Generated descriptive statistics (mean, std, min, max, quartiles) for numerical columns, providing insights into their distribution.



df.shape: Determined the number of rows and columns.



df.columns: Listed all column names.



3\. Data Cleaning



Missing Values: Verified for missing values using df.isnull().sum(). No missing values were found.



Duplicate Values: Checked for duplicate rows using 

df.duplicated().sum(). No duplicates were found, so df.drop\_duplicates(inplace=True) had no effect but was included as a good practice.



Outlier Detection and Removal (Price):



Outliers in the 'price' column were identified using the Interquartile Range (IQR) method.

Q1 (25th percentile) and Q3 (75th percentile) were calculated.

IQR was computed as Q3 - Q1.

Lower and upper bounds were defined as Q1 - 1.5 \* IQR and Q3 + 1.5 \* IQR respectively.

Rows where 'price' fell outside these bounds were removed, effectively cleaning extreme price values from the dataset.



4\. Descriptive Statistics (Post-Cleaning)



After outlier removal, the following statistics were recalculated for numerical columns:

df.mean(): Average values.

df.median(): Middle values.

df.mode().iloc\[0]: Most frequent values.

df.var(): Variance.

df.std(): Standard deviation.

df.describe(): A complete statistical summary of the cleaned numerical data.

Categorical Value Counts: value\_counts() was used to inspect the distribution of unique values in 'cut', 'color', and 'clarity' columns.



5\. Correlation Analysis



Correlation Matrix: df.corr(numeric\_only=True) computed the pairwise correlation between numerical columns.



Heatmap Visualization: A heatmap was generated using seaborn.heatmap() to visually represent the correlations, making it easier to identify strong positive or negative relationships.



6\. Data Visualization



Price Distribution: A histogram of 'price' was plotted using plt.hist() to show its frequency distribution.



Carat vs. Price: A scatter plot of 'carat' against 'price' was created using plt.scatter() to visualize their relationship.

Diamond Cut Distribution: A bar plot of df\['cut'].value\_counts() was used to display the distribution of different diamond cuts.



Pairplot: A seaborn.pairplot() was generated for 'carat', 'depth', 'table', and 'price' to show relationships between pairs of these numerical variables, along with their individual distributions.

#### 

#### Tools Used

pandas: For data manipulation and analysis.

numpy: For numerical operations (implicitly used by pandas).

matplotlib.pyplot: For basic plotting and visualization.

seaborn: For advanced statistical data visualization.

