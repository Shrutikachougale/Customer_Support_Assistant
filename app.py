import joblib
import streamlit as st
import google.generativeai as genai

# Load the pre-trained classifier and fitted TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")  # Fitted TF-IDF Vectorizer
classifier = joblib.load("query_classifier.pkl")  # Trained classifier

def classify_query(query):
    query_tfidf = vectorizer.transform([query])  # Ensure it's fitted before using transform()
    return classifier.predict(query_tfidf)[0]

# Google Gemini API Setup
genai.configure(api_key="your_key")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def get_gemini_response(query):
    category = classify_query(query)
    prompt = f"You are an AI customer support assistant. A customer asked: '{query}'. Category: {category}. Provide a direct response."
    response = model.generate_content(prompt)
    return f"Category: {category}\n\nResponse: {response.text}"

# Streamlit UI
st.title("💬 AI Customer Support Assistant")
query = st.text_input("Enter your query:")
if st.button("Get Response"):
    if query:
        response = get_gemini_response(query)
        st.write(response)
    else:
        st.warning("Please enter a query.")
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize and fit the vectorizer again
vectorizer = TfidfVectorizer()
queries = ["Where is my order?", "How do I return an item?", "What are the product details?",
           "I need help with my account", "How do I cancel my order?"]
vectorizer.fit(queries)  # Fit vectorizer before using transform()
