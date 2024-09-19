# 1. Title and Author
   # Project Title: **Sentiment Analysis and Quality Metrics in Amazon Product Reviews**

   Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
 
   Author Name: Yeshwanth Rayankula
 
   Link to the author's GitHub repo of the project: https://github.com/yeshwanthrayankula11/UMBC-DATA606-Capstone/tree/main
 
   Link to the author's LinkedIn profile: https://www.linkedin.com/in/yeshwanth-rayankula-3b798a17a/


# 2.  Project Background

  -   What is it about?
  -   This project aims to analyze Amazon product reviews to understand customer sentiment, extract insights, and identify key entities using traditional machine learning techniques and deep learning models, including BERT.

  -   Why does it matter?
  -   Understanding customer sentiment is crucial for businesses to improve product quality and customer satisfaction. Analyzing reviews provides insights into customer preferences and areas for improvement, ultimately boosting sales and fostering brand loyalty.

  -   What are your research questions?

  - What are the common sentiments expressed in Amazon product reviews?
  - How can we use machine learning models to predict product ratings based on review text?
  - What entities (products, brands, etc.) are frequently mentioned in reviews, and how do they relate to customer sentiment?
  - What sentiment patterns can be identified across different product categories (e.g., electronics, clothing, food) in Amazon reviews?

  Some specific questions related to this project:
  - How does the sentiment of a review correlate with the product rating given by the user? Is there a significant difference in sentiment between high-rated and low-rated reviews?
  - Can we identify common themes or topics within the text of the reviews that contribute to positive or negative sentiments? What are the top keywords associated with each sentiment?
  - What role does the length of the review play in determining the sentiment expressed? Are longer reviews more likely to be positive or negative?
  - How do helpfulness metrics (HelpfulnessNumerator and HelpfulnessDenominator) influence the perceived sentiment of reviews? Do more helpful reviews show different sentiment trends compared to less helpful ones?
  - Which specific entities (products, brands, or features) receive the most favorable or unfavorable sentiments, and how do these sentiments impact customer purchasing behavior?
  - How effective are different machine learning models (traditional and deep learning) in accurately predicting review scores based on sentiment analysis?
  - Can a recommendation system be developed to suggest products based on positive sentiment trends identified in user reviews? How does this compare to traditional recommendation approaches?



# 3. Data
  Describe the datasets you are using to answer your research questions.

  Data sources
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
  - Review_Length (length of the review in words)
  - HelpfulnessNumerator
  - HelpfulnessDenominator
