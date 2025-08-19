import pandas as pd
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocess import clean_text

def train(csv_path, artifacts_dir):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model Accuracy: {acc:.4f}")

    joblib.dump(model, f"{artifacts_dir}/model.pkl")
    joblib.dump(vectorizer, f"{artifacts_dir}/vectorizer.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--artifacts", type=str, required=True)
    args = parser.parse_args()
    train(args.csv, args.artifacts)
