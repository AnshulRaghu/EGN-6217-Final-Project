import pandas as pd

def preprocess(df):
    df = df.dropna()
    df['amount'] = df['amount'].astype(float)

    # Encode category (target)
    df['category_encoded'] = df['category'].astype('category').cat.codes

    return df