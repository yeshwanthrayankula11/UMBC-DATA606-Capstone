#!/usr/bin/env python
# coding: utf-8

# In[25]:


import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


# In[31]:


import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# In[34]:


# Path to your local fine-tuned model
MODEL_PATH = "./Bert_Model"


# Load the tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    return tokenizer, model


# In[35]:


# Load dataset for insights
@st.cache_data
def load_data():
    data_path = "./updated_dataset.csv" 
    data = pd.read_csv(data_path)
    return data


# In[36]:


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



# In[37]:


# Streamlit App
st.title("Sentiment Analysis with DistilBERT")
st.write("Analyze the sentiment of text reviews using a fine-tuned DistilBERT model.")


# In[38]:


# Load the model and tokenizer
tokenizer, model = load_model_and_tokenizer()


# In[ ]:





# In[39]:


# Sidebar for dataset exploration
st.sidebar.title("Explore Dataset")
if st.sidebar.button("Load Data"):
    data = load_data()
    st.sidebar.write(data.head())
    st.sidebar.bar_chart(data['SentimentLabel'].value_counts())  

# Input for sentiment analysis
st.header("Input a Review for Sentiment Analysis")
user_input = st.text_area("Enter a review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, tokenizer, model)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a valid review text.")


# In[ ]:





# In[ ]:




