### Tools Used:



pandas: For data loading, manipulation, and cleaning (import pandas as pd).



matplotlib.pyplot: For data visualization (histograms, scatter plots, line plots - import matplotlib.pyplot as plt).



seaborn: For enhanced data visualization (count plots - import seaborn as sns).



sklearn.preprocessing.LabelEncoder: For encoding categorical features (from sklearn.preprocessing import LabelEncoder).



sklearn.preprocessing.StandardScaler: For scaling numerical features (from sklearn.preprocessing import StandardScaler).



sklearn.model\_selection.train\_test\_split: For splitting data into training and testing sets (from sklearn.model\_selection import train\_test\_split).



sklearn.linear\_model.LogisticRegression: For building a Logistic Regression model (from sklearn.linear\_model import LogisticRegression).



sklearn.tree.DecisionTreeClassifier: For building a Decision Tree Classifier model (from sklearn.tree import DecisionTreeClassifier).



sklearn.metrics.accuracy\_score: For evaluating model performance (from sklearn.metrics import accuracy\_score).



Process and Steps Taken:



Data Loading:

The passenger data.csv file was loaded into a pandas DataFrame (df=pd.read\_csv("/content/passenger data.csv")).



Initial Data Exploration:

The first few rows of the DataFrame were displayed (df.head()).

Data types and non-null counts were inspected (df.info()).

Descriptive statistics were generated (df.describe()).



Column Identification:

Numerical and categorical columns were identified using df.select\_dtypes.



Handling Missing Values:

Missing values for each column were counted (df.isnull().sum()).

Missing Age values were imputed with the median age (df\["Age"] = df\["Age"].fillna(df\["Age"].median())).



Missing Embarked values were filled with the mode (most frequent value) (df\["Embarked"] = df\["Embarked"].fillna(df\["Embarked"].mode()\[0])).



Missing Cabin values were filled with the string "Unknown" (df\["Cabin"] = df\["Cabin"].fillna("Unknown")).



Missing values were re-checked to confirm successful handling.



Outlier Treatment (Fare Column):

The first quartile (Q1), third quartile (Q3), and Interquartile Range (IQR) were calculated for the Fare column.



Lower and upper bounds for outlier detection were determined using the 1.5 \* IQR rule.

Rows where the Fare was outside these bounds were removed from the DataFrame.





Duplicate Handling:

Duplicate rows were checked (df.duplicated().sum()).

df.drop\_duplicates(inplace=True) was used to ensure no duplicates remained.



Exploratory Data Analysis (EDA) - Visualizations:

A count plot was generated for Survived to show the distribution of survivors vs. non-survivors (sns.countplot(x="Survived", data=df)).

A count plot showed Pclass distribution, separated by Survived status (sns.countplot(x="Pclass", hue="Survived", data=df)).



A count plot showed Sex distribution, separated by Survived status (sns.countplot(x="Sex", hue="Survived", data=df)).



A histogram displayed the distribution of Age (plt.hist(df\["Age"], bins=20)).



A scatter plot visualized the relationship between Age and Fare (plt.scatter(df\["Age"], df\["Fare"])).



A line plot of Age was generated (plt.plot(df\["Age"])).



Feature Engineering / Preprocessing:

Label Encoding: The Sex and Embarked categorical columns were converted into numerical representations using LabelEncoder (df\["Sex"] = le.fit\_transform(df\["Sex"]), df\["Embarked"] = le.fit\_transform(df\["Embarked"])).



Feature Scaling: The numerical Age and Fare columns were standardized using StandardScaler (df\[\["Age", "Fare"]] = scaler.fit\_transform(df\[\["Age", "Fare"]])).

Column Dropping: Irrelevant columns including PassengerId, Name, Ticket, and Cabin were dropped from the DataFrame (df.drop(\["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)).





Model Preparation:

The dataset was split into features (X = df.drop("Survived", axis=1)) and the target variable (y = df\["Survived"]).

The data was further divided into training (80%) and testing (20%) sets using train\_test\_split with a random\_state for reproducibility.



Model Training and Evaluation:

Logistic Regression: A LogisticRegression model was initialized, trained on the training data, and used to predict Survived on the test set. Its accuracy\_score was then calculated and printed.



Decision Tree Classifier: A DecisionTreeClassifier model was initialized, trained, used for predictions on the test set, and its accuracy\_score was calculated and printed.



Model Comparison: The accuracy scores of both models were compared, and a statement indicating which model performed better was printed.

