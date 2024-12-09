import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# Path to your local fine-tuned model
MODEL_PATH = os.path.abspath("./Bert_Model")

# Load the tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
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

# Base dataset analysis
@st.cache_data
def load_base_dataset():
    try:
        dataset_path = "updated_dataset.csv"  # Update with the actual dataset path
        data = pd.read_csv(dataset_path)
        if 'Cleaned_Text' not in data.columns or 'PredictedSentiment' not in data.columns:
            st.error("Base dataset must have 'Cleaned_Text' and 'PredictedSentiment' columns.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading base dataset: {e}")
        return None

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Analyze sentiments using an existing dataset, individual review, or upload a custom dataset.")

# Load the model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# Option Selection
option = st.sidebar.selectbox("Choose an Option", ["Base Dataset Analysis", "Individual Review", "Upload Dataset"])

if option == "Base Dataset Analysis":
    st.header("Base Dataset Sentiment Analysis")
    base_data = load_base_dataset()
    if base_data is not None:
        st.write("Base Dataset Preview:")
        st.write(base_data.head())

        sentiment_counts = base_data['PredictedSentiment'].value_counts()
        st.write("Sentiment Counts:")
        st.bar_chart(sentiment_counts)

elif option == "Individual Review":
    st.header("Input a Review for Sentiment Analysis")
    user_input = st.text_area("Enter a review:", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input, tokenizer, model)
            st.success(f"The predicted sentiment is: **{sentiment}**")
        else:
            st.warning("Please enter a valid review text.")

elif option == "Upload Dataset":
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
            st.error(f"Error processing the uploaded dataset: {e}")
