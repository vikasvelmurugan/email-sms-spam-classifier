import streamlit as st

import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
stopword = stopwords.words('english')
s = string.punctuation

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopword and i not in s:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("EMAIL/SMS Spam classifier by Vikas")
input_sms=st.text_area("Enter the Message:")

if st.button('Predict'):
    # 1. preprocess

    transformed_sms = transform_text(input_sms)

    # 2. vectorize

    vector_input=tfidf.transform([transformed_sms])
    # 3. predict

    result=model.predict(vector_input)[0]
    # 4. display

    if result == 1:
        st.header("Spam!!!")
    else:
        st.header("Not Spam :)")



