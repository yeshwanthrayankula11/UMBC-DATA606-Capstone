import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        raise e

# Predict sentiment
def predict_sentiment(review, tokenizer, model):
    tokens = tokenizer.encode_plus(
        review,
        max_length=512,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class]

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of text reviews and generate predictions for a dataset.")

# Load the model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# Single review sentiment analysis
st.header("Single Review Analysis")
user_input = st.text_area("Enter a review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, tokenizer, model)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a valid review text.")

# Dataset upload and sentiment prediction
st.header("Upload Dataset for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview:")
        st.write(data.head())

        # Check for the required text column
        if 'Cleaned_Text' in data.columns:
            st.write("Generating sentiment predictions...")
            data['PredictedSentiment'] = data['Cleaned_Text'].apply(lambda x: predict_sentiment(x, tokenizer, model))
            st.write("Processed Dataset with Sentiments:")
            st.dataframe(data)

            # Allow downloading the updated dataset
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv",
            )
        else:
            st.error("The uploaded dataset must contain a 'Cleaned_Text' column.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
