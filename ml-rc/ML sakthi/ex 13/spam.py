import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("spam.csv", delimiter=',', header=None, encoding='ISO-8859-1')  # Change delimiter if needed
print("DataFrame shape:", df.shape)
print("Columns in DataFrame:", df.columns)
if df.shape[1] < 2:
    raise ValueError("The DataFrame does not have enough columns. Check the data format.")
print(df.describe())
print(df.dtypes)
print(df.head())
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0], random_state=42)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
classifier = LogisticRegression(solver='liblinear') 
classifier.fit(X_train, y_train)
X_test = vectorizer.transform(['URGENT! Your Mobile No 1234 was awarded a Prize', 'Hey honey, whats up?'])
predictions = classifier.predict(X_test)
print("Result:")
print(predictions)


