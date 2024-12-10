import os
import gdown
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Directory to store the downloaded model
MODEL_DIR = "drive_bert_model"

# Google Drive file IDs
MODEL_FILES = {
    "tokenizer.json": "1KM7-jZAtlif-tBoj6bSkbSkHdcsZdZog",
    "model.safetensors": "1Pv4pMPucrf40YgTu06NFpVZr1LxnGjyS",
    "config.json": "1XbHNfFeoH5aWOk442J5BUkkFlA4A-Fiw",
    "tokenizer_config.json": "1-aZiDSN5jUIwsZA2Hh7TlGpnYQ61iSAQ",
    "special_tokens_map.json": "1-aZiDSN5jUIwsZA2Hh7TlGpnYQ61iSAQ",
    "vocab.txt": "1BU-HHhbEHP9p3Ltb1hD7m7ufSNnDFexd",
}

# Dataset Google Drive file ID
DATASET_FILE_ID = "1tqkp-LpDAgVDFlfdzgCJ5eYJjpkACr2W"

# Function to download model files
def download_model_files():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    for filename, file_id in MODEL_FILES.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            gdown.download(url, output_path, quiet=False)

# Function to download the dataset
def download_dataset():
    dataset_path = "updated_dataset.csv"
    if not os.path.exists(dataset_path):
        url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
        print("Downloading dataset...")
        gdown.download(url, dataset_path, quiet=False)
    return dataset_path

# Load the model and tokenizer
@st.cache_resource
def load_model():
    download_model_files()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load the dataset
@st.cache_data
def load_data():
    dataset_path = download_dataset()
    data = pd.read_csv(dataset_path)
    return data

# Main Streamlit app
def main():
    st.title("Amazon Product Review Sentiment Analysis")
    
    # Tabs
    tabs = st.tabs(["Dataset Summary", "Insights", "Sentiment Prediction"])
    
    # Tab 1: Dataset Summary
    with tabs[0]:
        st.header("Dataset Summary")
        try:
            data = load_data()
            st.write("Dataset Sample:")
            st.write(data.head())
            st.write("Dataset Statistics:")
            st.write(data.describe())
        except FileNotFoundError:
            st.error("Dataset file not found. Please check the file path.")
    
    # Tab 2: Insights
    with tabs[1]:
        st.header("Insights")
        try:
            data = load_data()
            
            # Visualization 1: Score Distribution
            st.subheader("Score Distribution")
            if 'Score' in data.columns:
                fig, ax = plt.subplots()
                data['Score'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_xlabel("Score")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.warning("The dataset does not contain a 'Score' column.")
        except FileNotFoundError:
            st.error("Dataset file not found. Please check the file path.")

    # Tab 3: Sentiment Prediction
    with tabs[2]:
        st.header("Sentiment Prediction")
        try:
            sentiment_pipeline = load_model()
            
            review_text = st.text_area("Enter a review to analyze:")
            if st.button("Analyze"):
                if review_text.strip():
                    result = sentiment_pipeline(review_text)
                    sentiment = result[0]['label']
                    confidence = result[0]['score']
                    st.write(f"**Sentiment**: {sentiment}")
                    st.write(f"**Confidence**: {confidence:.2f}")
                else:
                    st.warning("Please enter a valid review.")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
