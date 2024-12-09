import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import gdown
from safetensors.torch import load_file  # Import for handling .safetensors

# Define constants
MODEL_PATH = os.path.abspath("./Bert_Model")
MODEL_FILE = os.path.join(MODEL_PATH, "model.safetensors")
MODEL_FILE_ID = "1Pv4pMPucrf40YgTu06NFpVZr1LxnGjyS"  # Google Drive ID for safetensors model file

# Download the model file if not present
def download_model():
    """Download the safetensors model file from Google Drive if not already present."""
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model file from Google Drive...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_FILE, quiet=False)

# Load the tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    """Load the tokenizer and model, downloading files as needed."""
    try:
        # Ensure model file is downloaded
        download_model()

        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

        # Load model using safetensors loader
        state_dict = load_file(MODEL_FILE)  # Load state_dict from safetensors
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            state_dict=state_dict,  # Pass the state_dict to the model
            local_files_only=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        raise e

# Predict sentiment
def predict_sentiment(review, tokenizer, model):
    """Predict the sentiment of a single review."""
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
st.header("Input a Review for Sentiment Analysis")
user_input = st.text_area("Enter a review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, tokenizer, model)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a valid review text.")

# Dataset upload and sentiment prediction
st.sidebar.title("Classify Uploaded Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Preview:")
    st.write(data.head())

    # Check for the required text column
    if 'Cleaned_Text' in data.columns:
        # Apply sentiment prediction to each review in the Cleaned_Text column
        st.write("Generating sentiment predictions...")
        data['PredictedSentiment'] = data['Cleaned_Text'].apply(lambda x: predict_sentiment(x, tokenizer, model))
        st.write("Processed Dataset with Sentiments:")
        st.dataframe(data)

        # Allow downloading the updated dataset
        st.download_button("Download Results", data.to_csv(index=False), "results.csv")
    else:
        st.error("The uploaded dataset must contain a 'Cleaned_Text' column.")
