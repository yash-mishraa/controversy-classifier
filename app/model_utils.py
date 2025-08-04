import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def load_data(path: str):
    return pd.read_csv(path)

def extract_features(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    analyzer = SentimentIntensityAnalyzer()

    embeddings = model.encode(texts)
    sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]
    features = np.hstack([embeddings, np.array(sentiments).reshape(-1, 1)])
    return features

def train_and_save_model(csv_path: str, output_model_path: str):
    df = load_data(csv_path)
    X = extract_features(df["text"].tolist())
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    clf.fit(X_train, y_train)

    print("Model accuracy on test set:", clf.score(X_test, y_test))

    joblib.dump(clf, output_model_path)
    print(f"✅ Model saved to: {output_model_path}")

if __name__ == "__main__":
    train_and_save_model("data/tweets.csv", "models/model.pkl")
