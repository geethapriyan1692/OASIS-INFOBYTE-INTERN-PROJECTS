## Spam Detection

This project focuses on detecting spam messages using natural language processing (NLP) and machine learning techniques. The dataset used is a collection of SMS messages labeled as either 'ham' (non-spam) or 'spam'.

## Table of Contents

Installation
Data Overview
Data Preprocessing
Exploratory Data Analysis
Text Preprocessing
Feature Engineering
Modeling
Evaluation
Contributing

## Installation
To get started with this project, clone the repository and install the necessary dependencies:
numpy
pandas
matplotlib
seaborn
wordcloud
nltk
scikit-learn
joblib


## Data Preprocessing
The data preprocessing steps include handling missing values, removing unnecessary columns, and encoding categorical variables.


import pandas as pd

df = pd.read_csv("C:/Users/PRIYAN/Downloads/spam.csv", encoding='latin1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.columns = ["Target", "Message"]
df["Target"].replace("ham", 0, inplace=True)
df["Target"].replace("spam", 1, inplace=True)

## Exploratory Data Analysis
Visualize the data to understand the distribution of spam and ham messages.
import matplotlib.pyplot as plt

# Pie chart
plt.pie(df["Target"].value_counts(), labels=["Ham", "Spam"], autopct="%.2f")
plt.show()

# Bar plot
X = ["Ham", "Spam"]
Y = df["Target"].value_counts()
plt.bar(X, Y, color=["pink", "cyan"])
plt.title("Ratio of Spam and Ham messages")
plt.show()

# Word clouds
from wordcloud import WordCloud

spam_text = str(df[df["Target"] == 1]["Message"])
ham_text = str(df[df["Target"] == 0]["Message"])

# Generate word clouds
spam_wordcloud = WordCloud().generate(spam_text)
ham_wordcloud = WordCloud().generate(ham_text)

# Display word clouds
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.show()

plt.imshow(ham_wordcloud)
plt.axis('off')
plt.show()

# Text Preprocessing

Preprocess the text data by removing URLs, punctuations, stopwords, and performing stemming.


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download("all")

def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in word_tokenize(text)])
    return text

df["transformed"] = df["Message"].apply(transform)


## Feature Engineering

Convert text data into numerical features using Bag of Words and TF-IDF.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Bag of Words
cv = CountVectorizer()
X_bow = cv.fit_transform(df["transformed"]).toarray()
joblib.dump(cv, "count_vectorizer.pkl")

# TF-IDF
tf = TfidfVectorizer()
X_tfidf = tf.fit_transform(df["transformed"]).toarray()
joblib.dump(tf, "tfidf_vectorizer.pkl")

## Feature Engineering
Convert text data into numerical features using Bag of Words and TF-IDF.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Bag of Words
cv = CountVectorizer()
X_bow = cv.fit_transform(df["transformed"]).toarray()
joblib.dump(cv, "count_vectorizer.pkl")

# TF-IDF
tf = TfidfVectorizer()
X_tfidf = tf.fit_transform(df["transformed"]).toarray()
joblib.dump(tf, "tfidf_vectorizer.pkl")

## Modeling
Train different Naive Bayes models to classify the messages.

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, df["Target"], test_size=0.20, random_state=0)

# Multinomial Naive Bayes
mnb_model = MultinomialNB()
mnb_model.fit(X_train, Y_train)
joblib.dump(mnb_model, "MNB_model.pkl")

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)

# Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_model.fit(X_train, Y_train)



Here's a detailed README file for the provided Python script, formatted for a GitHub repository:

Spam Detection
This project focuses on detecting spam messages using natural language processing (NLP) and machine learning techniques. The dataset used is a collection of SMS messages labeled as either 'ham' (non-spam) or 'spam'.

Table of Contents
Installation
Data Overview
Data Preprocessing
Exploratory Data Analysis
Text Preprocessing
Feature Engineering
Modeling
Evaluation
Contributing
License
Installation
To get started with this project, clone the repository and install the necessary dependencies:

bash
Copy code
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install --upgrade pip
pip install -r requirements.txt
requirements.txt should include:

text
Copy code
numpy
pandas
matplotlib
seaborn
wordcloud
nltk
scikit-learn
joblib
Data Overview
The dataset contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. The data file spam.csv has the following columns:

Target: The label of the message (ham or spam)
Message: The content of the SMS message
Data Preprocessing
The data preprocessing steps include handling missing values, removing unnecessary columns, and encoding categorical variables.

python
Copy code
import pandas as pd

df = pd.read_csv("C:/Users/PRIYAN/Downloads/spam.csv", encoding='latin1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.columns = ["Target", "Message"]
df["Target"].replace("ham", 0, inplace=True)
df["Target"].replace("spam", 1, inplace=True)
Exploratory Data Analysis
Visualize the data to understand the distribution of spam and ham messages.

python
Copy code
import matplotlib.pyplot as plt

# Pie chart
plt.pie(df["Target"].value_counts(), labels=["Ham", "Spam"], autopct="%.2f")
plt.show()

# Bar plot
X = ["Ham", "Spam"]
Y = df["Target"].value_counts()
plt.bar(X, Y, color=["pink", "cyan"])
plt.title("Ratio of Spam and Ham messages")
plt.show()

# Word clouds
from wordcloud import WordCloud

spam_text = str(df[df["Target"] == 1]["Message"])
ham_text = str(df[df["Target"] == 0]["Message"])

# Generate word clouds
spam_wordcloud = WordCloud().generate(spam_text)
ham_wordcloud = WordCloud().generate(ham_text)

# Display word clouds
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.show()

plt.imshow(ham_wordcloud)
plt.axis('off')
plt.show()
Text Preprocessing
Preprocess the text data by removing URLs, punctuations, stopwords, and performing stemming.

python
Copy code
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download("all")

def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in word_tokenize(text)])
    return text

df["transformed"] = df["Message"].apply(transform)
Feature Engineering
Convert text data into numerical features using Bag of Words and TF-IDF.

python
Copy code
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Bag of Words
cv = CountVectorizer()
X_bow = cv.fit_transform(df["transformed"]).toarray()
joblib.dump(cv, "count_vectorizer.pkl")

# TF-IDF
tf = TfidfVectorizer()
X_tfidf = tf.fit_transform(df["transformed"]).toarray()
joblib.dump(tf, "tfidf_vectorizer.pkl")
Modeling
Train different Naive Bayes models to classify the messages.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, df["Target"], test_size=0.20, random_state=0)

# Multinomial Naive Bayes
mnb_model = MultinomialNB()
mnb_model.fit(X_train, Y_train)
joblib.dump(mnb_model, "MNB_model.pkl")

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)

# Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_model.fit(X_train, Y_train)

## Evaluation
Evaluate the models using precision and confusion matrix.

from sklearn.metrics import precision_score, confusion_matrix
import seaborn as sns

# Evaluate Bernoulli Naive Bayes
Y_pred_bnb = bnb_model.predict(X_test)
precision_bnb = precision_score(Y_test, Y_pred_bnb)
print(f"Bernoulli Naive Bayes Precision: {precision_bnb}")

# Confusion matrix for Bernoulli Naive Bayes
cm_bnb = confusion_matrix(Y_test, Y_pred_bnb)
sns.heatmap(cm_bnb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluate Multinomial Naive Bayes
Y_pred_mnb = mnb_model.predict(X_test)
precision_mnb = precision_score(Y_test, Y_pred_mnb)
print(f"Multinomial Naive Bayes Precision: {precision_mnb}")

# Confusion matrix for Multinomial Naive Bayes
cm_mnb = confusion_matrix(Y_test, Y_pred_mnb)
sns.heatmap(cm_mnb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


Here's a detailed README file for the provided Python script, formatted for a GitHub repository:

Spam Detection
This project focuses on detecting spam messages using natural language processing (NLP) and machine learning techniques. The dataset used is a collection of SMS messages labeled as either 'ham' (non-spam) or 'spam'.

Table of Contents
Installation
Data Overview
Data Preprocessing
Exploratory Data Analysis
Text Preprocessing
Feature Engineering
Modeling
Evaluation
Contributing
License
Installation
To get started with this project, clone the repository and install the necessary dependencies:

bash
Copy code
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install --upgrade pip
pip install -r requirements.txt
requirements.txt should include:

text
Copy code
numpy
pandas
matplotlib
seaborn
wordcloud
nltk
scikit-learn
joblib
Data Overview
The dataset contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. The data file spam.csv has the following columns:

Target: The label of the message (ham or spam)
Message: The content of the SMS message
Data Preprocessing
The data preprocessing steps include handling missing values, removing unnecessary columns, and encoding categorical variables.

python
Copy code
import pandas as pd

df = pd.read_csv("C:/Users/PRIYAN/Downloads/spam.csv", encoding='latin1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.columns = ["Target", "Message"]
df["Target"].replace("ham", 0, inplace=True)
df["Target"].replace("spam", 1, inplace=True)
Exploratory Data Analysis
Visualize the data to understand the distribution of spam and ham messages.

python
Copy code
import matplotlib.pyplot as plt

# Pie chart
plt.pie(df["Target"].value_counts(), labels=["Ham", "Spam"], autopct="%.2f")
plt.show()

# Bar plot
X = ["Ham", "Spam"]
Y = df["Target"].value_counts()
plt.bar(X, Y, color=["pink", "cyan"])
plt.title("Ratio of Spam and Ham messages")
plt.show()

# Word clouds
from wordcloud import WordCloud

spam_text = str(df[df["Target"] == 1]["Message"])
ham_text = str(df[df["Target"] == 0]["Message"])

# Generate word clouds
spam_wordcloud = WordCloud().generate(spam_text)
ham_wordcloud = WordCloud().generate(ham_text)

# Display word clouds
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.show()

plt.imshow(ham_wordcloud)
plt.axis('off')
plt.show()
Text Preprocessing
Preprocess the text data by removing URLs, punctuations, stopwords, and performing stemming.

python
Copy code
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download("all")

def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in word_tokenize(text)])
    return text

df["transformed"] = df["Message"].apply(transform)
Feature Engineering
Convert text data into numerical features using Bag of Words and TF-IDF.

python
Copy code
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Bag of Words
cv = CountVectorizer()
X_bow = cv.fit_transform(df["transformed"]).toarray()
joblib.dump(cv, "count_vectorizer.pkl")

# TF-IDF
tf = TfidfVectorizer()
X_tfidf = tf.fit_transform(df["transformed"]).toarray()
joblib.dump(tf, "tfidf_vectorizer.pkl")
Modeling
Train different Naive Bayes models to classify the messages.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, df["Target"], test_size=0.20, random_state=0)

# Multinomial Naive Bayes
mnb_model = MultinomialNB()
mnb_model.fit(X_train, Y_train)
joblib.dump(mnb_model, "MNB_model.pkl")

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)

# Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_model.fit(X_train, Y_train)
Evaluation
Evaluate the models using precision and confusion matrix.

python
Copy code
from sklearn.metrics import precision_score, confusion_matrix
import seaborn as sns

# Evaluate Bernoulli Naive Bayes
Y_pred_bnb = bnb_model.predict(X_test)
precision_bnb = precision_score(Y_test, Y_pred_bnb)
print(f"Bernoulli Naive Bayes Precision: {precision_bnb}")

# Confusion matrix for Bernoulli Naive Bayes
cm_bnb = confusion_matrix(Y_test, Y_pred_bnb)
sns.heatmap(cm_bnb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## Contributing

Here's a detailed README file for the provided Python script, formatted for a GitHub repository:

Spam Detection
This project focuses on detecting spam messages using natural language processing (NLP) and machine learning techniques. The dataset used is a collection of SMS messages labeled as either 'ham' (non-spam) or 'spam'.

Table of Contents
Installation
Data Overview
Data Preprocessing
Exploratory Data Analysis
Text Preprocessing
Feature Engineering
Modeling
Evaluation
Contributing
License
Installation
To get started with this project, clone the repository and install the necessary dependencies:

bash
Copy code
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install --upgrade pip
pip install -r requirements.txt
requirements.txt should include:

text
Copy code
numpy
pandas
matplotlib
seaborn
wordcloud
nltk
scikit-learn
joblib
Data Overview
The dataset contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. The data file spam.csv has the following columns:

Target: The label of the message (ham or spam)
Message: The content of the SMS message
Data Preprocessing
The data preprocessing steps include handling missing values, removing unnecessary columns, and encoding categorical variables.

python
Copy code
import pandas as pd

df = pd.read_csv("C:/Users/PRIYAN/Downloads/spam.csv", encoding='latin1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.columns = ["Target", "Message"]
df["Target"].replace("ham", 0, inplace=True)
df["Target"].replace("spam", 1, inplace=True)
Exploratory Data Analysis
Visualize the data to understand the distribution of spam and ham messages.

python
Copy code
import matplotlib.pyplot as plt

# Pie chart
plt.pie(df["Target"].value_counts(), labels=["Ham", "Spam"], autopct="%.2f")
plt.show()

# Bar plot
X = ["Ham", "Spam"]
Y = df["Target"].value_counts()
plt.bar(X, Y, color=["pink", "cyan"])
plt.title("Ratio of Spam and Ham messages")
plt.show()

# Word clouds
from wordcloud import WordCloud

spam_text = str(df[df["Target"] == 1]["Message"])
ham_text = str(df[df["Target"] == 0]["Message"])

# Generate word clouds
spam_wordcloud = WordCloud().generate(spam_text)
ham_wordcloud = WordCloud().generate(ham_text)

# Display word clouds
plt.imshow(spam_wordcloud)
plt.axis('off')
plt.show()

plt.imshow(ham_wordcloud)
plt.axis('off')
plt.show()
Text Preprocessing
Preprocess the text data by removing URLs, punctuations, stopwords, and performing stemming.

python
Copy code
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download("all")

def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in word_tokenize(text)])
    return text

df["transformed"] = df["Message"].apply(transform)
Feature Engineering
Convert text data into numerical features using Bag of Words and TF-IDF.

python
Copy code
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

# Bag of Words
cv = CountVectorizer()
X_bow = cv.fit_transform(df["transformed"]).toarray()
joblib.dump(cv, "count_vectorizer.pkl")

# TF-IDF
tf = TfidfVectorizer()
X_tfidf = tf.fit_transform(df["transformed"]).toarray()
joblib.dump(tf, "tfidf_vectorizer.pkl")
Modeling
Train different Naive Bayes models to classify the messages.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, df["Target"], test_size=0.20, random_state=0)

# Multinomial Naive Bayes
mnb_model = MultinomialNB()
mnb_model.fit(X_train, Y_train)
joblib.dump(mnb_model, "MNB_model.pkl")

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)

# Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_model.fit(X_train, Y_train)
Evaluation
Evaluate the models using precision and confusion matrix.

python
Copy code
from sklearn.metrics import precision_score, confusion_matrix
import seaborn as sns

# Evaluate Bernoulli Naive Bayes
Y_pred_bnb = bnb_model.predict(X_test)
precision_bnb = precision_score(Y_test, Y_pred_bnb)
print(f"Bernoulli Naive Bayes Precision: {precision_bnb}")

# Confusion matrix for Bernoulli Naive Bayes
cm_bnb = confusion_matrix(Y_test, Y_pred_bnb)
sns.heatmap(cm_bnb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluate Multinomial Naive Bayes
Y_pred_mnb = mnb_model.predict(X_test)
precision_mnb = precision_score(Y_test, Y_pred_mnb)
print(f"Multinomial Naive Bayes Precision: {precision_mnb}")

# Confusion matrix for Multinomial Naive Bayes
cm_mnb = confusion_matrix(Y_test, Y_pred_mnb)
sns.heatmap(cm_mnb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.
