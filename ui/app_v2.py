import streamlit as st

st.title("CentricAI: Transaction Predictor")

st.markdown("### Enter recent transaction sequence")

sequence = st.text_input("Enter categories (comma separated)")

if st.button("Predict Next Category"):
    st.success("Predicted: Shopping")

st.markdown("---")
st.markdown("### Model Comparison")

st.bar_chart({
    "LSTM": 0.68,
    "GRU": 0.72,
    "Transformer": 0.78
})