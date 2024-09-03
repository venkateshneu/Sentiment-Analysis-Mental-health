import streamlit as st
import pandas as pd
import numpy as np
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load the data
df = pd.read_csv("mental_health.csv")

# Preprocess text
def text_preprocessing(text):
    stemmer = PorterStemmer()
    text = re.sub(r'https?://[^\s/$.?#].[^\s]*', '', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    processed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    processed_text = " ".join(processed_tokens)
    return processed_text

df["text"] = df["text"].apply(text_preprocessing)

# Train/Test split
from sklearn.model_selection import train_test_split
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred = nb_model.predict(X_test_tfidf)

# Streamlit App
st.title("Mental Health Text Classification")

st.write("""
### This app classifies text data into mental health categories.
""")

user_input = st.text_area("Enter a text for classification", "")

if user_input:
    processed_input = text_preprocessing(user_input)
    input_tfidf = tfidf.transform([processed_input])
    prediction = nb_model.predict(input_tfidf)
    prediction_label = "Positive (1)" if prediction[0] == 1 else "Negative (0)"
    
    st.write(f"**Prediction:** {prediction_label}")
    
    # Show model performance metrics
    st.write("### Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    
    # Show classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Show confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    st.pyplot(plt)

