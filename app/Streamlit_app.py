
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

# Load pre-trained BERT model and tokenizer
def load_model(model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Predict sentiment for user input
def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(predictions, dim=1).item()
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[sentiment]

# Load Dataset and Display Stats
def load_and_display_stats(dataset_path):
    data = pd.read_csv(dataset_path)
    st.write("Dataset loaded successfully.")
    st.write("Class Distribution:")
    st.write(data['Sentiment'].value_counts())
    return data

# Streamlit app interface
def main():
    st.title("Sentiment Analysis App")
    st.write("Analyze the sentiment of your reviews using a fine-tuned BERT model.")

    # Directory and file paths
    model_dir = os.path.join("app", "bert_model_folder")
    dataset_path = os.path.join("app", "updated_dataset", "cleaned_dataset.csv")

    # Load model and tokenizer
    st.write("Loading model...")
    model, tokenizer = load_model(model_dir)
    st.write("Model loaded successfully.")

    # Display dataset statistics
    if st.checkbox("Show Dataset Stats"):
        load_and_display_stats(dataset_path)

    # User input for sentiment prediction
    user_input = st.text_area("Enter your review:", "")
    if st.button("Predict Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(model, tokenizer, user_input)
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Please enter a review to predict its sentiment.")

if __name__ == "__main__":
    main()
