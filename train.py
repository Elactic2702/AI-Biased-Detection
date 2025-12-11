# src/train.py (minimal)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

df = pd.read_csv('data/processed/resumes_clean.csv')
X = df['text']
y = df['label']

pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=20000)),
                 ('clf', LogisticRegression(max_iter=200))])
pipe.fit(X, y)
joblib.dump(pipe, 'artifacts/logreg_baseline.joblib')
