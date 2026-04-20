class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)