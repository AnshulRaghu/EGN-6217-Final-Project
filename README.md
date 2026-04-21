# 💰 CentricAI: Transaction Sequence Prediction System

## 📌 Overview
**CentricAI** is a deep learning-powered prototype designed to predict the **next transaction category** based on a user's recent sequence of financial activity. Developed as part of the EGN-6217 course, this project focuses on comparing state-of-the-art sequence modeling architectures to identify which best captures the temporal dependencies of spending patterns.

Rather than modeling specific individuals, the system learns general short-term transaction behaviors from sequential data, providing a comparative framework for **LSTM**, **GRU**, and **Transformer** models.

---

## 🧠 System Purpose
The primary goal is to provide a comparative analysis of deep learning models in the personal finance domain:
* **Predictive Modeling:** Forecast the next transaction category given a sliding window of historical transactions.
* **Architectural Comparison:** Evaluate and compare the performance of:
    * **LSTM** (Long Short-Term Memory)
    * **GRU** (Gated Recurrent Unit)
    * **Transformer** (Self-Attention based)
* **Interactive Testing:** Provide a Streamlit interface for real-time model interaction and result visualization.

---

## 🗂️ Repository Structure
```text
EGN-6217-Final-Project/
│
├── data/        # Raw and processed transaction datasets
├── notebooks/   # Training, evaluation, and model comparison (train_models.ipynb)
├── results/     # Accuracy plots, loss curves, and saved model weights
├── src/         # Core logic: preprocessing, model definitions, and utility scripts
├── ui/          # Streamlit web interface (app_v2.py)
│
├── requirements.txt
└── Deliverables/ # Project reports and IEEE documentation
