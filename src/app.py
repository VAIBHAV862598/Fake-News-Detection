import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Dataset

data = {
"text": [
# REAL
"ISRO successfully launches new satellite",
"Indian government announces new education reforms",
"Stock market closes at record high",
"Ajit Pawar is dead",
"Supreme Court delivers verdict on new law",
"NASA prepares for Artemis moon mission",
# FAKE
"Aliens landed in Delhi yesterday",
"Actor confirms he is a time traveler",
"Scientists prove earth is flat",
"Ajit Pawar turns into robot during speech",
"Drinking petrol cures all diseases",
"Moon will crash into Earth tomorrow"
],
"label": [1,1,1,1,1,1,0,0,0,0,0,0]
}

df = pd.DataFrame(data)

# ML Pipeline

pipeline = Pipeline([
("tfidf", TfidfVectorizer(stop_words="english")),
("model", LogisticRegression())
])

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
df["text"], df["label"], test_size=0.2, random_state=42
)

# Train Model

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

# Streamlit UI

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below to check if it is Real or Fake.")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# Input

user_input = st.text_area("Enter News Text")

# Button Logic (FIXED INDENTATION)


if user_input.strip() == "":
    st.warning("Please enter some news text.")

else:
    prediction = pipeline.predict([user_input])[0]
    probability = pipeline.predict_proba([user_input])[0]

    if prediction == 1:
        st.success("This News is REAL (" + str(round(probability[1]*100, 2)) + "% confidence)")
    else:
        st.error("This News is FAKE (" + str(round(probability[0]*100, 2)) + "% confidence)")

st.markdown("---")
st.write("Developed using Streamlit & Machine Learning")