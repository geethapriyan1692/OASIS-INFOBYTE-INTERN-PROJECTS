# CAR PRICE PREDICTION:

This project aims to analyze and predict car prices based on various features such as car specifications, condition, and market trends. The dataset used for this project is stored in a CSV file named car data.csv.

# TABLE OF CONTENTS :
Installation
Data Overview
Data Preprocessing
Exploratory Data Analysis
Feature Engineering
Modeling
Evaluation
Contribution

# INSTALLATION:
seaborn
lazypredict
pandas
numpy
plotly
matplotlib
scikit-learn
xgboost
lightgbm

# DATA OVERVIEW:

The dataset contains various features about cars, including:

Year
Selling_Price
Present_Price
Driven_kms
Car_Name
Fuel_Type
Selling_type
Transmission
Owner

# DATA PREPROCESSING:

The data preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical features.

df = pd.read_csv("C:/Users/PRIYAN/Downloads/car data.csv")

# Checking for missing values
df.isnull().sum()

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
df['Car_Name'] = encode.fit_transform(df['Car_Name'])
df['Fuel_Type'] = encode.fit_transform(df['Fuel_Type'])
df['Selling_type'] = encode.fit_transform(df['Selling_type'])
df['Transmission'] = encode.fit_transform(df['Transmission'])
df['Owner'] = encode.fit_transform(df['Owner'])

# Standardization using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# EXPLORATRY DATA ANALYSIS: 
Visualize the data to understand the relationships between variables.

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
correlation_matrix = df.corr()
fig = px.imshow(correlation_matrix, color_continuous_scale='Cividis', title="Correlation Matrix")
fig.show()

# Distribution plots
sns.boxplot(data=df[['Year', 'Driven_kms', 'Present_Price']])
sns.histplot(data=df, x='Year', bins=20, kde=True)
sns.regplot(data=df, x='Selling_Price', y='Driven_kms')
sns.barplot(data=df, x='Selling_type', y='Selling_Price')
sns.boxplot(data=df, x='Transmission', y='Selling_Price')

# FEATURE ENGINEERING

# Example of feature engineering
df['log_Selling_Price'] = np.log(df['Selling_Price'] + 1)

# MODELING

Train different regression models to predict car prices.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

class Regressor_models:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def decisiontree(self):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor().fit(self.x_train, self.y_train)
        y_pred_train = model.predict(self.x_train)
        y_pred_test = model.predict(self.x_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        return {'R2_train': r2_train, 'R2_test': r2_test}

    # Add other models similarly...

model = Regressor_models(X_train, X_test, y_train, y_test)
accuracy_scores = {
    'DecisionTree': model.decisiontree(),
    # Add other models...
}

# EVALUATION 
Evaluate the performance of the models using metrics such as R2 score.

print(accuracy_scores)

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.

