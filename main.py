import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



nltk.download('stopwords', download_dir=r'C:\Users\nada\AppData\Roaming\nltk_data')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

data = pd.read_csv(r"C:\Users\nada\OneDrive\Desktop\SentimentScope\IMDB Dataset.csv")
# print(data.head())
# print(data.columns)
# print(data.info())
# print(data.isnull().sum())
# print(data['sentiment'].value_counts())

# nltk.download('stopword')
# stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['clean_review'] = data['review'].apply(clean_text)
print(data[['review','clean_review']].head())

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_review'])
print(X.shape)

y = data['sentiment'].map({'positive': 1, 'negative': 0})
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training", X_train.shape[0])
print("Testing", X_test.shape[0])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text_vector = tfidf.transform([text])
    pred = model.predict(text_vector)[0]
    return "Positive " if pred == 1 else "Negative "

print(predict_sentiment("I really loved this movie, it was amazing!"))
print(predict_sentiment("The film was boring and too long."))

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Saved")

loaded_model = joblib.load("sentiment_model.pkl")
loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment_loaded(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text_vector = loaded_tfidf.transform([text])
    pred = loaded_model.predict(text_vector)[0]
    return "Positive " if pred == 1 else "Negative "

print(predict_sentiment_loaded("I hate this movie, it was terrible!"))

