# member1_preprocess.py (snippet)
import pandas as pd
df = pd.read_csv("data/raw/resumes.csv")
# simple PII scrub
df['text'] = df['resume_text'].fillna('').str.replace(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b','[NAME]', regex=True)
# bias table
bias_table = df.groupby('gender').agg(count=('id','count'), positive_rate=('label', 'mean')).reset_index()
bias_table.to_csv('reports/bias_table.csv', index=False)
