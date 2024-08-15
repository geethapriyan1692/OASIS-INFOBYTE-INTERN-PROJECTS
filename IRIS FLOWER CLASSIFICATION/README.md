## Iris Classification Project
This project focuses on the classification of the Iris dataset using various machine learning algorithms. The dataset consists of 150 samples of iris flowers, each with four features: Sepal Length, Sepal Width, Petal Length, and Petal Width. The goal is to classify the flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica.## IRIS FLOWER CLASSIFICATION

## CONTENTS :
Installation
Dataset
Exploratory Data Analysis (EDA)
Data Preprocessing
Model Training and Evaluation
Results
Contributing

## Installation
To run this project, you need to have Python installed along with the following libraries:

pandas
matplotlib
seaborn
scikit-learn
numpy


## Dataset
The Iris dataset can be downloaded from here. It is also included in this repository as Iris.csv.


## Exploratory Data Analysis (EDA)
## Importing Libraries and Dataset
python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Initial Data Inspection
df.head()
df.info()

## Data Distribution
Histogram for Sepal Length:
plt.figure(figsize=(9, 5))
sns.histplot(df['SepalLengthCm'], kde=True, bins=10)
plt.title('Distribution of Sepal Length')
plt.xlabel('SepalLengthCm')
plt.ylabel('Frequency')
plt.show()

## Box Plot for Sepal Length by Species:

plt.figure(figsize=(9, 5))
sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

## Pair Plot for All Features:
sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
plt.show()

## Heatmap for Correlation Between Features:
plt.figure(figsize=(9, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidth=0.5)
plt.title('Correlation Heatmap')
plt.show()

## Scatter Plot for Sepal Length vs Sepal Width Colored by Species:
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', s=80, data=df)
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()

## Violin Plot for Petal Length vs Species:

plt.figure(figsize=(9, 5))
sns.violinplot(x='Species', y='PetalLengthCm', data=df)
plt.title('Petal Length vs Species')
plt.xlabel('Species')
plt.ylabel('PetalLengthCm')
plt.show()

## Data Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x = df.drop(columns=['Species'])
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

## Model Training and Evaluation
## SVM

from sklearn import svm, metrics

model = svm.SVC()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy of the SVM is:", metrics.accuracy_score(prediction, y_test))

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy score of the Logistic Regression is:", metrics.accuracy_score(prediction, y_test))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
prediction = DT.predict(x_test)
print("Accuracy score of the Decision Tree Classifier is:", metrics.accuracy_score(prediction, y_test))

## K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Accuracy score of the KNN is:", metrics.accuracy_score(prediction, y_test))

## Results
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

print('Confusion Matrix:')
cn = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 6))
plt.imshow(cn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
