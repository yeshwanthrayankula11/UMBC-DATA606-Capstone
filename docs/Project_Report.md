# 1. Title and Author
   # Project Title: **Sentiment Analysis and Quality Metrics in Amazon Product Reviews**
   - ![image](https://github.com/user-attachments/assets/b7f0a93f-102b-4b28-be10-c09db2d44bce)

   - Author Name: Rayankula Yeshwanth
   - Semester: FALL 2024
   - Prepared for: UMBC Data Science Master Degree Capstone by Dr. Chaojie Wang
   - GitHub Repository: [Checkout my Github Repo](https://github.com/yeshwanthrayankula11/UMBC-DATA606-Capstone)
   - LinkedIn Profile: [Checkout my linkedIn Profile](https://www.linkedin.com/in/yeshwanth-rayankula-3b798a17a/)
   - PowerPoint Presentation: 
   - Streamlit App: [Visit the Sentiment Analysis App](https://umbc-data606-capstone-8zcuwcpdfrkqq8j5653hmw.streamlit.app/)
   - YouTube Video: 
   
# 2.  Project Background

  -  What is it about?
  -  This project aims to analyze Amazon product reviews to understand customer sentiment, extract insights, and identify key entities using traditional machine learning techniques and deep learning 
     models, including BERT.

  - Why does it matter?
  - Understanding customer sentiment is crucial for businesses to improve product quality and customer satisfaction. Analyzing reviews provides insights into customer preferences and areas for 
    improvement, ultimately boosting sales and fostering brand loyalty.

  -   What are your research questions?

  - What are the common sentiments expressed in Amazon product reviews?
  - How can we use machine learning models to predict product ratings based on review text?
  - What entities (products, brands, etc.) are frequently mentioned in reviews, and how do they relate to customer sentiment?
  - What sentiment patterns can be identified across different product categories (e.g., electronics, clothing, food) in Amazon reviews?

  Some specific questions related to this project:
  - How does the sentiment of a review correlate with the product rating given by the user? Is there a significant difference in sentiment between high-rated and low-rated reviews?
  - Can we identify common themes or topics within the text of the reviews that contribute to positive or negative sentiments? What are the top keywords associated with each sentiment?
  - How do helpfulness metrics (HelpfulnessNumerator and HelpfulnessDenominator) influence the perceived sentiment of reviews? Do more helpful reviews show different sentiment trends compared to less           helpful ones?
  - How effective are different machine learning models (traditional and deep learning) in accurately predicting review scores based on sentiment analysis?


# 3. Data
  ##  Data sources:
  
  - The dataset is sourced from Amazon, and contains customer reviews for various products.
  - Data size: 301 MB
  - Data shape: 567,092 rows and 10 columns
  - Time-period: The dataset does not specify a time range, but the reviews span multiple years based on product release dates.
  
  ## What does each row represent?
   - Each row represents an individual product review submitted by a customer.
  
  ## Data dictionary:
   <img width="544" alt="image" src="https://github.com/user-attachments/assets/fde5f921-c650-41bf-b9a3-339403ccf67a">
  
  
  Definition
  
  Which variable/column will be your target/label in your ML model?
  - The target/label will be the Score column (1 to 5 stars).
  
  Which variables/columns may be selected as features/predictors for your ML models?
  - Cleaned_Text (text of the review)
  - Sentiment
  - Review_Length (length of the review in words)
  - HelpfulnessNumerator
  - HelpfulnessDenominator


# 4 Data Cleaning

   ## Missing Value Handling:
   - Identified missing values using .isnull().sum().
   - Dropped irrelevant columns ('Id', 'ProfileName') and rows with missing 'Summary'.
   
   ## Duplicate Removal:
   - Identified duplicate rows based on specific columns like 'ProductId' and 'Text'.
   - Removed duplicates using .drop_duplicates() and rechecked for duplicates.
   
   ## Balancing the Dataset:
   - Used the minimum count of reviews across all rating categories to create a balanced dataset.
   - Grouped by 'Score' and sampled uniformly to ensure equal representation.
   
   ## Data Cleaning and Transformation:
   - Removal of Tags, URLs, Punctuations, lowercasing of words, stopword removal etc,.
   - Ensured that words like no, not, nor, like, good, great etc,. are not removed which might affect the sentiment.
   - Removed specific words (e.g., "one") from text columns using string replacement.
   - Verified the shape and missing values after cleaning.


# 5 Data Exploration & Feature Extraction Steps
   
   ## Basic Exploration:
   - Reviewed the dataset using .head(), .info(), and .describe().
   - Converted time data to a human-readable datetime format and verified the conversion.
   
   ## Visualization of Score Distribution:
   - Created a count plot and annotated it with percentages.
   - Explored balanced dataset distributions using bar plots and pie charts.
   
   ## Top Products Analysis:
   - Identified the top 10 products by review count and visualized them with a bar plot.
   
   ## Feature Engineering:
   - Created a HelpfulnessRatio feature by dividing helpful votes by total votes.
   - Examined its distribution using histograms.
   
   ## Outlier Detection:
   - Analyzed review lengths by computing interquartile range (IQR).
   - Filtered out extreme values and re-plotted review lengths without outliers.
   
   ## Temporal Analysis:
   - Plotted the number of reviews over time to observe trends.
   - Calculated and visualized the average review score per year.
   
   ## Helpfulness Ratio vs. Score:
   - Compared helpfulness ratios across different scores using box plots.

### Saved the cleaned Dataset for easy access and further modeling and analysis.

# 6 Modeling:

## Traditional ML Models:

### Logistic Regression:
- Performed logistic regression and the evaluation report is as follows:
- <img width="441" alt="image" src="https://github.com/user-attachments/assets/7bea0831-5eee-401d-9eb5-e55b9fa2c4e5">


### SVM:
- Performed SVM and the evaluation report is as follows:
- <img width="412" alt="image" src="https://github.com/user-attachments/assets/184acfe4-3e02-401c-9c52-8cdb487efe7c">

### Random Forest:
- Performed Random Forest and the evaluation report is as follows:
- <img width="416" alt="image" src="https://github.com/user-attachments/assets/884e0a14-ecc8-4cf9-aeab-436e5fa2c226">


## NLP model - Pretrained Distilled BERT
- Performed Pretrained Distilled BERT and the evaluation report is as follows:
- <img width="351" alt="image" src="https://github.com/user-attachments/assets/3e36f4b8-7b13-4423-86c1-1ddc4ea5d8db">
- <img width="410" alt="image" src="https://github.com/user-attachments/assets/2e70fb42-9ca5-48fd-a409-ce2c51105dbb">



### So far, the BERT model has been the best performaing model compared to the rest of the models.
### So, I chose this to move forward for my application.
- To do so, I saved the model and uploaded it to a google drive folder which can be openly accessible to anyone with the link.
- Also, the dataset size is around 100 mb. So, for easy access and due to the size limitations of github(25mb), I uploaded it to my google drive folder and downloaded the dataset dynamically similar to the bert model..
- Bert_Model: https://drive.google.com/drive/folders/1EBY6HEPcaskIhYsQA00yP9aZL8Zdzly4?usp=drive_link.
- Dataset: https://drive.google.com/file/d/1tqkp-LpDAgVDFlfdzgCJ5eYJjpkACr2W/view?usp=drive_link.

# 7 Topic Modelling and Aspect Based Analysis.

## Steps and Techniques
### Data Preprocessing
- Input Data: Reviews are preprocessed to remove stopwords, convert to lowercase, and tokenize.
- Sampling: A subset of the dataset is used for computational efficiency.

### Topic Modeling
- Method: Latent Dirichlet Allocation (LDA).
- Steps:
   - Convert reviews into a document-term matrix using CountVectorizer.
   - Extract latent topics with the top contributing words.
- Visualize topics with bar charts.
- Tools: scikit-learn, gensim.

### Aspect-Based Sentiment Analysis
- Tools:
   - SpaCy for extracting aspects (noun phrases).
   - VADER sentiment analysis for polarity scoring.
- Steps:
   - Extract aspects from reviews.
   - Calculate sentiment scores for aspects.
   - Aggregate and display top positive and negative aspects.

### Visualization
- Word Clouds:
   - Positive and negative aspects are visualized using contrasting color maps.
- Bar Charts:
   - Top contributing words for each topic.
   - Sentiment distribution across extracted aspects.
   - Insights
-  Topic Modeling:
   - Key themes in customer reviews are uncovered, helping understand product quality, features, and user experience.
- Sentiment Analysis:
   - Aspects like "delivery", "packaging", and "price" show varying sentiment trends.
   - Clear differentiation between positively and negatively perceived product attributes.
- Word Clouds:
   - Positive words (e.g., "excellent", "fast delivery").
   - ![image](https://github.com/user-attachments/assets/c4842ba5-27db-421d-a14a-df7600e97946)

   - Negative words (e.g., "poor quality", "late shipping").
   - ![image](https://github.com/user-attachments/assets/6bfa0db9-d669-438a-a41d-f8563b0b64ba)

 
## Some key visualizations are as follows:
- ![image](https://github.com/user-attachments/assets/e66b052e-73b5-4b71-ac1b-15fcb7534d54)
- <img width="466" alt="image" src="https://github.com/user-attachments/assets/6ef070db-307c-42e6-93ce-83d459537bea">
- <img width="529" alt="image" src="https://github.com/user-attachments/assets/f5ff1b13-0a7b-497a-aa71-cb35278e084c">


# 8 Streamlit Application Deployment:

# 9 Conclusion:



 





    


   
