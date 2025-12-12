import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

class MLP(nn.Module):
    """Simple PyTorch MLP classifier for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, epochs=10, batch_size=256, lr=1e-3):
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")

def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    return probs

def evaluate_model(model, X_test, y_test):
    y_prob = predict_proba(model, X_test)
    auc = roc_auc_score(y_test, y_prob)

    # Best threshold for F1
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.9, 41):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return auc, best_f1, best_t
