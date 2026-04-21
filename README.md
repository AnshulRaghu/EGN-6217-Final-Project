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
```

---
## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/AnshulRaghu/EGN-6217-Final-Project.git]
cd EGN-6217-Final-Project
```

---

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

🚀 Running the Pipeline

▶️ Option 1: Training and Evaluation

To retrain the models or view the detailed evaluation metrics (Accuracy, F1-Score), run the main notebook:
```bash
jupyter notebook notebooks/train_models.ipynb
```

🖥️ Option 2: Launch the Interface

To test predictions in a user-friendly environment, launch the Streamlit app:
```bash
streamlit run ui/app_v2.py
```

📊 Performance Summary
The following metrics represent the final evaluation of each model on the test dataset:

| Model       | Accuracy | F1 Score |
| ----------- | -------- | -------- |
| LSTM        | ~0.70    | ~0.69    |
| GRU         | ~0.73    | ~0.72    |
| Transformer | ~0.76    | ~0.75    |

📈 Key Insights
* Transformer Dominance: The Transformer model achieved the highest performance, showcasing its ability to capture broad contextual relationships within transaction sequences better than recurrent architectures
* GRU vs. LSTM: GRU slightly outperformed LSTM while maintaining higher computational efficiency (fewer parameters)

⚠️ Known Issues & Limitations
* ❗ Lack of User IDs: The current dataset does not distinguish between individual users; sequences represent general population patterns
* ❗ Feature Constraints: Models currently rely solely on category sequences. Factors like transaction amount and time-of-day are not yet integrated
* ❗ Dataset Size: Results are based on a limited academic dataset; performance may vary with larger-scale, real-world data
* ❗ Minimal UI: The interface is a proof-of-concept for demonstration and lacks production-grade error handling

⚠️ Usage Warnings
* This system is built for **academic and research purposes only**
* Not Financial Advice: Predictions should never be used as a basis for real-world financial decisions or budgeting

---

📬 Contact Anshul Raghuvanshi
* 🔗 GitHub: https://github.com/AnshulRaghu
* 📧 Email: anshul2raghu@gmail.com
* Developed for EGN-6217: Deep Learning Final Project
