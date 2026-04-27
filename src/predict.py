import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path

# Get project root (go up from src/ to Project/)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "saved_models" / "lstm_model.pth"
ENCODER_PATH = BASE_DIR / "saved_models" / "label_encoder.pkl"

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(
            32,
            len(label_encoder.classes_)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Load model
model = LSTMModel()
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)
model.eval()


def predict_next_category(cat1, cat2, cat3, cat4, cat5):
    categories = [cat1, cat2, cat3, cat4, cat5]

    encoded = label_encoder.transform(categories)

    X = np.array(encoded).reshape(1, 5, 1)
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X)

        probabilities = torch.softmax(output, dim=1)

        pred_idx = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            pred_idx
        ].item()

    predicted_category = label_encoder.inverse_transform(
        [pred_idx]
    )[0]

    return predicted_category, confidence