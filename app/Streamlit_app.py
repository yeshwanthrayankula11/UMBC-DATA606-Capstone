import streamlit as st
import gdown
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Google Drive file IDs
model_file_id = "1Pv4pMPucrf40YgTu06NFpVZr1LxnGjyS"  
tokenizer_file_id = "1KM7-jZAtlif-tBoj6bSkbSkHdcsZdZog"  

# Check and download the model
if not os.path.exists("bert_model.pt"):
    download_file_from_google_drive(model_file_id, "bert_model.pt")

# Check and download the tokenizer
if not os.path.exists("tokenizer"):
    os.makedirs("tokenizer", exist_ok=True)
    download_file_from_google_drive(tokenizer_file_id, "tokenizer/tokenizer.zip")
    os.system("unzip tokenizer/tokenizer.zip -d tokenizer")

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_model.pt")
tokenizer = BertTokenizer.from_pretrained("tokenizer")

# Streamlit app UI
st.title("Sentiment Analysis App")
st.subheader("I can classify your review into Positive, Neutral, or Negative.")

# Input from user
user_input = st.text_area("Enter a review:")

# Predict sentiment
if st.button("Classify"):
    if user_input.strip():
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_index = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][sentiment_index].item()

        # Map sentiment index to label
        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_labels[sentiment_index]

        # Display results
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Level:** {confidence * 100:.2f}%")
    else:
        st.write("Please enter some text to classify.")
