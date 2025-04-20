
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def load_and_clean_data():
    # Load datasets
    df_fake = pd.read_csv('data/Fake.csv')
    df_real = pd.read_csv('data/True.csv')

    # Label datasets
    df_fake['label'] = 0
    df_real['label'] = 1

    # Combine and shuffle
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title and text
    df['content'] = df['title'] + " " + df['text']
    df['content'] = df['content'].apply(clean_text)

    # Features and labels
    X = df['content']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
