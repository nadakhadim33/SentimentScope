# SentimentScope

**SentimentScope** is a **Sentiment Analysis project** using **NLP and Machine Learning** to determine whether a text (review) is positive or negative.  
The project is built with Python and libraries including Pandas, NLTK, Scikit-learn, and Joblib.

---

## Features
- Text preprocessing: cleaning reviews, removing punctuation and stopwords  
- Converting text into numerical data using **TF-IDF**  
- Training a **Logistic Regression** model for sentiment prediction  
- Evaluating the model with Accuracy, Precision, Recall, and Confusion Matrix  
- Predicting sentiment for new sentences in real-time  
- Saving the trained model and TF-IDF vectorizer for future use  

---

## Dataset
- **IMDB Movie Reviews Dataset**  
- Contains 50,000 movie reviews (positive and negative)  
- Can be downloaded from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

---

## Installation
1. Install required libraries:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib
