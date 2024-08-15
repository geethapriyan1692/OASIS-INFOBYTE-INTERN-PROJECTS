# Sales Prediction 

## Project Overview
This project focuses on predicting sales based on advertising spend in different media such as TV, Radio, and Newspaper. The dataset used in this project is the "Advertising" dataset, which contains data on the advertising spend in these media and the corresponding sales figures. The goal is to build various regression models to predict sales and evaluate their performance.

## Table of Contents
Project Overview
Data Description
Exploratory Data Analysis (EDA)
Data Preprocessing
Model Training and Evaluation
Conclusion
Installation and Usage

## Data Description
The dataset consists of the following columns:

TV: Advertising spend on TV
Radio: Advertising spend on Radio
Newspaper: Advertising spend on Newspaper
Sales: Sales figures
The data is read from a CSV file and the initial steps involve checking for missing values, duplicate rows, and obtaining a statistical summary of the dataset.

## Exploratory Data Analysis (EDA)
EDA includes visualizing the distribution of features and their relationships:

Histograms: To understand the distribution of TV, Radio, Newspaper, and Sales.
Boxplots: To detect outliers in the data.
Correlation Heatmap: To visualize the correlation between different features.

# Data Preprocessing
Handling missing values and duplicate rows.
Dropping irrelevant columns.
Handling outliers using Interquartile Range (IQR).
Splitting the data into training and testing sets.
Standardizing the features using StandardScaler.

# Model Training and Evaluation
Various regression models are trained and evaluated for their performance:

Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
AdaBoost Regressor
Extra Trees Regressor
Each model is evaluated using the R-squared score for both training and testing sets.

## Conclusion
The models are compared based on their performance metrics, and it is concluded that Extra Trees Regressor, Gradient Boosting, and AdaBoost are the most optimal models for this dataset.
