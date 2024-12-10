# Streamlit Application for Sentiment analysis of Amazon product reviews and aspect based topic modelling

## Features
- Sentiment prediction for individual reviews.
- Dataset analysis with sentiment and confidence scores.
- Interactive visualizations for insights.
- Downloadable processed dataset with predictions.

## Technologies Used
* Python
* Streamlit
* Transformers (Hugging Face)
* Matplotlib
* Pandas


## The app code considers of the following aspects:
- **Dataset Summary**: The app has the dataset preview and the statistics for the dataset.
- **Insights**: It shows few visualizations.
- **Sentiment Prediction**: It gives you chance for two types of uploads:
  - Single review input typing it manually. Output is the sentiment label with confidence.
  - Upload a file sizing upto 200mb.
  - It creates an output with an additional columns that gives labels predicting the sentiment(positive, negative, or neutral) and confidence of the prediction accuracy.
  - There is an option to download these results for your dashboards or for your presentations or reports.
