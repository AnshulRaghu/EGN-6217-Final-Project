import sys
import os
from pathlib import Path
import pickle
import gradio as gr

# ----------------------------
# Project root setup
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import predict_next_category

# ----------------------------
# Paths
# ----------------------------
ENCODER_PATH = BASE_DIR / "saved_models" / "label_encoder.pkl"

# ----------------------------
# Load label encoder
# ----------------------------
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

category_choices = list(label_encoder.classes_)

# ----------------------------
# Core function
# ----------------------------
def advisor(c1, c2, c3, c4, c5):
    category, confidence = predict_next_category(c1, c2, c3, c4, c5)

    result = f"""
Predicted Next Spending Category: {category}

Confidence Score: {confidence * 100:.2f}%

Financial Recommendation:
"""

    if category == "Food & Drink":
        result += "Frequent food spending detected. Consider setting a weekly dining budget."

    elif category == "Entertainment":
        result += "Entertainment expenses may rise. Track non-essential purchases carefully."

    elif category == "Utilities":
        result += "Recurring bills expected. Prepare fixed monthly expenses early."

    else:
        result += "Maintain balanced spending habits and continue tracking expenses."

    return result

# ----------------------------
# Gradio UI
# ----------------------------
demo = gr.Interface(
    fn=advisor,
    inputs=[
        gr.Dropdown(choices=category_choices, label="Previous Transaction 1"),
        gr.Dropdown(choices=category_choices, label="Previous Transaction 2"),
        gr.Dropdown(choices=category_choices, label="Previous Transaction 3"),
        gr.Dropdown(choices=category_choices, label="Previous Transaction 4"),
        gr.Dropdown(choices=category_choices, label="Previous Transaction 5"),
    ],
    outputs="text",
    title="CentricAI - AI Spending Pattern Predictor",
    description="Predicts next spending category using LSTM-based sequence modeling."
)

demo.launch()