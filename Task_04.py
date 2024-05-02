import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("kaggle/twitter_training.csv")

# Check the first few rows of the dataset
print(data.head())

# Analyze target distribution
target_counts = data['target'].value_counts()

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=data, palette='viridis')
plt.title('target Distribution in Social Media Data')
plt.xlabel('target')
plt.ylabel('Count')
plt.show()

######################################################################################
import numpy as np
import pandas as pd 
import os
import re
import spacy
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print(stopwords.words('english'))
l=stopwords.words('english')
data_train=pd.read_csv("kaggle/twitter_training.csv")
data_test=pd.read_csv("kaggle/twitter_validation.csv")

data_train.head()
data_train.shape

col=['Tweet ID',
'entity',
'target',
'Tweet']
data_train.columns =col
data_train.head()

df=pd.DataFrame(data_train)
df.head()
df.info()

df=df[df['Tweet'].str.len()!=0] 
# this is very important point to handle the empty tweet

df['Tweet'] = df['Tweet'].astype(str)
df['Tweet'] = df['Tweet'].str.lower()  # Apply lower() to each element in the 'Tweet' column
print(df['Tweet'])
print(df.head())

import en_core_web_sm
nlp=spacy.load("en_core_web_sm")
def preprocess(text):
    doc=nlp(text)
    new_text=[]
    for word in doc:
        if word.is_stop or word.is_punct or word.text=="im":
            continue
        new_text.append(word.lemma_)
    return " ".join(new_text)

print(df['target'].value_counts())

df['stemmed_content']=df['Tweet'].apply(preprocess)
print(df.head())
# print(preprocess("there is a king who is ruling india for the longst server of all time"))


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
print(df.head())


print(df['target'].value_counts())
# 0-->Irrelevant 
# 1--> Negative 
# 2-->Positive
# 3-->Neutral

X=df['stemmed_content']
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=2)

print(X.shape,X_train.shape,X_test.shape,y.shape,y_train.shape,y_test.shape)

vectorizer=TfidfVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test = vectorizer.transform(X_test)
print(X_train)

from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier object
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

X_test_predict=model.predict(X_test)
X_test_acc=accuracy_score(y_test,X_test_predict)
print(X_test_acc)

X_test_predict
