import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ISLP import load_data
from torchmetrics import Metric
from pytorch_lightning import Trainer
from ISLP.torch import ErrorTracker, SimpleModule

# Custom Metric for tracking predictions within 10% range
class CorrectPredictionPercentage(Metric):
    def __init__(self, threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        correct = torch.abs((preds - target) / target) < self.threshold
        self.correct += correct.sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

# Load and prepare the data
NYSE = load_data('NYSE')
cols = ['DJ_return', 'log_volume', 'log_volatility']
X = pd.DataFrame(StandardScaler().fit_transform(NYSE[cols]), columns=NYSE[cols].columns, index=NYSE.index)
Y = NYSE['log_volume']

# Function to create rolling window sequences
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i: i + window_size])
    return np.array(sequences)

# Prepare sequences with window size
window_size = 5
X_seq = create_sequences(X.to_numpy(), window_size)
Y_seq = Y[window_size-1:].to_numpy()

# Split the data into train, validation, and test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

# Convert data to tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define LSTM model
class NYSELSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(NYSELSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        val, (h_n, c_n) = self.lstm(x)
        val = self.dense(self.dropout(val[:, -1]))
        return torch.flatten(val)

# Initialize model
input_size = X_seq.shape[2]
nyse_lstm_model = NYSELSTMModel(input_size=input_size)
nyse_optimizer = RMSprop(nyse_lstm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Instantiate the custom accuracy metric
accuracy_metric = CorrectPredictionPercentage(threshold=0.1)

# Lists to track the losses and custom accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Custom training loop with custom accuracy tracking
for epoch in range(10):  
    # Training phase
    nyse_lstm_model.train()
    running_train_loss = 0.0
    accuracy_metric.reset()  # Reset metric for the new epoch
    for X_batch, Y_batch in train_loader:
        nyse_optimizer.zero_grad()
        preds = nyse_lstm_model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        nyse_optimizer.step()
        running_train_loss += loss.item()
        accuracy_metric.update(preds, Y_batch)
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = accuracy_metric.compute().item()  # Calculate accuracy
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.6f}, Training Accuracy: {train_accuracy:.4f}")

    # Validation phase
    nyse_lstm_model.eval()
    running_val_loss = 0.0
    accuracy_metric.reset()  # Reset metric for validation
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            preds = nyse_lstm_model(X_batch)
            val_loss = criterion(preds, Y_batch)
            running_val_loss += val_loss.item()
            accuracy_metric.update(preds, Y_batch)
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy_metric.compute().item()  # Calculate accuracy
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.4f}")

# Final test phase to evaluate test loss and accuracy
nyse_lstm_model.eval()
running_test_loss = 0.0
accuracy_metric.reset()  # Reset metric for testing
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        preds = nyse_lstm_model(X_batch)
        test_loss = criterion(preds, Y_batch)
        running_test_loss += test_loss.item()
        accuracy_metric.update(preds, Y_batch)
final_test_loss = running_test_loss / len(test_loader)
final_test_accuracy = accuracy_metric.compute().item()

# Display test results
print("\nTest Metric              DataLoader 0")
print("----------------------------------------")
print(f"test_percent_correct     {final_test_accuracy:.6f}")

# Plot the training and validation losses over epochs
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()




