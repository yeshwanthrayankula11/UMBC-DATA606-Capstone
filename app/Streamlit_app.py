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
            import seaborn as sns
            
            # Visualization 1: Score Distribution
            st.subheader("Score Distribution with Percentages")
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(x=data['Score'], palette='viridis')
            total = float(len(data))
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 75, 
                        '{:1.1f} %'.format((height / total) * 100), 
                        ha="center", bbox=dict(facecolor='none', edgecolor='black', 
                                               boxstyle='round', linewidth=0.5))
            ax.set_title('Score Distribution (Ratings) with Percentages', fontsize=20, y=1.05)
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            sns.despine(right=True)
            st.pyplot(plt)

            # Visualization 2: Average Review Score Over Time
            st.subheader("Average Review Score Over Time")
            data['ReviewYear'] = pd.to_datetime(data['Time']).dt.year
            avg_score_by_year = data.groupby('ReviewYear')['Score'].mean()
            plt.figure(figsize=(10, 6))
            avg_score_by_year.plot()
            plt.title('Average Review Score Over Time')
            plt.xlabel('Year')
            plt.ylabel('Average Score')
            st.pyplot(plt)

            # Visualization 3: Number of Reviews Over Time
            st.subheader("Number of Reviews Over Time")
            reviews_over_time = data.groupby(data['ReviewYear']).size()
            plt.figure(figsize=(10, 6))
            reviews_over_time.plot()
            plt.title('Number of Reviews Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Reviews')
            st.pyplot(plt)

            # Visualization 5: Top Products by Review Count
            st.subheader("Top 10 Products by Review Count")
            top_products = data['ProductId'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_products.values, y=top_products.index, palette='coolwarm')
            plt.title('Top 10 Products by Review Count')
            plt.xlabel('Number of Reviews')
            plt.ylabel('Product ID')
            st.pyplot(plt)         

        except FileNotFoundError:
            st.error("Dataset file not found. Please check the file path.")

    # Sentiment Prediction

    with tabs[2]:
        st.header("Sentiment Prediction")
        try:
            # Load the sentiment analysis model pipeline
            sentiment_pipeline = load_model()

            # Individual Review Sentiment Prediction
            st.subheader("Analyze a Single Review")
            review_text = st.text_area("Enter a review to analyze:")
            if st.button("Analyze Review"):
                if review_text.strip():
                    result = sentiment_pipeline(review_text)
                    sentiment = result[0]['label']
                    confidence = result[0]['score']
                    st.write(f"**Sentiment**: {sentiment}")
                    st.write(f"**Confidence**: {confidence:.2f}")
                else:
                    st.warning("Please enter a valid review.")

            # Dataset Upload and Sentiment Prediction
            st.subheader("Analyze a Dataset")
            uploaded_file = st.file_uploader("Upload a CSV file with a 'Cleaned_Text' column", type="csv")
            if uploaded_file is not None:
                uploaded_data = pd.read_csv(uploaded_file)
                st.write("Uploaded Dataset Preview:")
                st.write(uploaded_data.head())

                if 'Cleaned_Text' in uploaded_data.columns:
                    st.write("Generating sentiment predictions...")

                    # Apply sentiment prediction for each row
                    def predict_with_confidence(text):
                        try:
                            result = sentiment_pipeline(text)
                            label = result[0]['label']  # Directly use the label from the pipeline
                            confidence = result[0]['score']
                            return label, confidence
                        except Exception as e:
                            return "Error", 0.0

                    uploaded_data[['PredictedSentiment', 'Confidence']] = uploaded_data['Cleaned_Text'].dropna().astype(str).apply(
                        lambda x: pd.Series(predict_with_confidence(x))
                    )

                    st.write("Processed Dataset with Sentiments and Confidence Scores:")
                    st.dataframe(uploaded_data)

                    # Allow downloading the updated dataset
                    csv = uploaded_data.to_csv(index=False)
                    st.download_button("Download Results", data=csv, file_name="results_with_sentiments.csv", mime="text/csv")
                else:
                    st.error("The uploaded dataset must contain a 'Cleaned_Text' column.")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")




if __name__ == "__main__":
    main()
