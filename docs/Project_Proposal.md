# 1. Title and Author
   # Project Title: **Sentiment Analysis and Quality Metrics in Amazon Product Reviews**

   Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
 
   Author Name: Yeshwanth Rayankula
 
   Link to the author's GitHub repo of the project: https://github.com/yeshwanthrayankula11/UMBC-DATA606-Capstone/tree/main
 
   Link to the author's LinkedIn profile: https://www.linkedin.com/in/yeshwanth-rayankula-3b798a17a/


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
  - What role does the length of the review play in determining the sentiment expressed? Are longer reviews more likely to be positive or negative?
  - How do helpfulness metrics (HelpfulnessNumerator and HelpfulnessDenominator) influence the perceived sentiment of reviews? Do more helpful reviews show different sentiment trends compared to less           helpful ones?
  - Which specific entities (products, brands, or features) receive the most favorable or unfavorable sentiments, and how do these sentiments impact customer purchasing behavior?
  - How effective are different machine learning models (traditional and deep learning) in accurately predicting review scores based on sentiment analysis?


# 3. Data
  Describe the datasets you are using to answer your research questions.

  Data sources:
  
  The dataset is sourced from Amazon, and contains customer reviews for various products.
  
  Data size: 301 MB
  
  Data shape: 567,092 rows and 10 columns
  
  Time-period: The dataset does not specify a time range, but the reviews span multiple years based on product release dates.
  
  What does each row represent?
  
  Each row represents an individual product review submitted by a customer.
  
  Data dictionary:
  
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

   Missing Value Handling:
   - Identified missing values using .isnull().sum().
   - Dropped irrelevant columns ('Id', 'ProfileName') and rows with missing 'Summary'.
   
   Duplicate Removal:
   - Identified duplicate rows based on specific columns like 'ProductId' and 'Text'.
   - Removed duplicates using .drop_duplicates() and rechecked for duplicates.
   
   Balancing the Dataset:
   - Used the minimum count of reviews across all rating categories to create a balanced dataset.
   - Grouped by 'Score' and sampled uniformly to ensure equal representation.
   
   Data Cleaning and Transformation:
   - Removal of Tags, URLs, Punctuations, lowercasing of words, stopword removal etc,.
   - Ensured that words like no, not, nor, like, good, great etc,. are not removed which might affect the sentiment.
   - Removed specific words (e.g., "one") from text columns using string replacement.
   - Verified the shape and missing values after cleaning.


# 5 Data Exploration & Feature Extraction Steps
   
   Basic Exploration:
      
   - Reviewed the dataset using .head(), .info(), and .describe().
   - Converted time data to a human-readable datetime format and verified the conversion.
   
   Visualization of Score Distribution:
   - Created a count plot and annotated it with percentages.
   - Explored balanced dataset distributions using bar plots and pie charts.
   
   Top Products Analysis:
   - Identified the top 10 products by review count and visualized them with a bar plot.
   
   Feature Engineering:
   - Created a HelpfulnessRatio feature by dividing helpful votes by total votes.
   - Examined its distribution using histograms.
   
   Outlier Detection:
   - Analyzed review lengths by computing interquartile range (IQR).
   - Filtered out extreme values and re-plotted review lengths without outliers.
   
   Temporal Analysis:
   - Plotted the number of reviews over time to observe trends.
   - Calculated and visualized the average review score per year.
   
   Helpfulness Ratio vs. Score:
   - Compared helpfulness ratios across different scores using box plots.

### Saved the cleaned Dataset for easy access and further modeling and analysis.

# Modeling:

## Traditional ML Models:

Logistic Regression:
- Performed logistic regression and the evaluation report is as follows:
- <img width="441" alt="image" src="https://github.com/user-attachments/assets/7bea0831-5eee-401d-9eb5-e55b9fa2c4e5">


SVM:
- Performed SVM and the evaluation report is as follows:
- <img width="412" alt="image" src="https://github.com/user-attachments/assets/184acfe4-3e02-401c-9c52-8cdb487efe7c">

Random Forest:
- Performed Random Forest and the evaluation report is as follows:
- <img width="416" alt="image" src="https://github.com/user-attachments/assets/884e0a14-ecc8-4cf9-aeab-436e5fa2c226">


## NLP model - Pretrained Distilled BERT
- Performed Pretrained Distilled BERT and the evaluation report is as follows:
- <img width="351" alt="image" src="https://github.com/user-attachments/assets/3e36f4b8-7b13-4423-86c1-1ddc4ea5d8db">
- <img width="410" alt="image" src="https://github.com/user-attachments/assets/2e70fb42-9ca5-48fd-a409-ce2c51105dbb">



# So far the BERT model has been the best performaing model compared to the rest of the models.




    


   
