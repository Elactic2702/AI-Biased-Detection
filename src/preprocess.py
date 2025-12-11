import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import os

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def clean_text(text):
    """Basic text cleaning: lowercase, remove special chars, remove extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# MAIN PREPROCESSING
# -----------------------------

def preprocess_data(input_path, output_path):
    print("🔍 Loading dataset...")
    df = pd.read_csv(input_path)

    print(f"📄 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean resume text
    print("🧹 Cleaning resume text...")
    df["cleaned_resume"] = df["resume_text"].apply(clean_text)

    # Encode gender as numbers
    print("🔢 Encoding gender...")
    gender_encoder = LabelEncoder()
    df["gender_encoded"] = gender_encoder.fit_transform(df["gender"])

    # Reorder columns
    df = df[["resume_text", "cleaned_resume", "gender", "gender_encoded", "label"]]

    # Save processed file
    print("💾 Saving processed file...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Preprocessing complete! File saved at: {output_path}")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    INPUT = "data/raw/resumes_sample.csv"
    OUTPUT = "data/processed/processed_data.csv"

    preprocess_data(INPUT, OUTPUT)
