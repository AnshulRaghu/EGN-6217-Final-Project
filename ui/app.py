import streamlit as st
import pandas as pd

st.title("CentricAI 💰")

uploaded_file = st.file_uploader("Upload transaction data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    st.write("Category Distribution")
    st.bar_chart(df['category'].value_counts())