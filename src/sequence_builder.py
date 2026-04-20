import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Ensure chronological order
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Encode category
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    return df, le

def create_sequences(df, seq_length=5):
    data = df['category_encoded'].values

    sequences = []
    labels = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)