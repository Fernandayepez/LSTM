from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
import torch
import torch.nn as nn
from torchmetrics import Metric
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate  # For final table display

# Define the CorrectPredictionPercentage metric
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

# Define the LSTM model with increased hidden size and dropout
class ACMTP10LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.3):
        super(ACMTP10LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        val, (h_n, c_n) = self.lstm(x)
        val = self.dense(self.dropout(val[:, -1]))
        return torch.flatten(val)

# LightningModule for training and testing
class LSTMModelModule(LightningModule):
    def __init__(self, input_size, lr=0.0001):  # Reduced learning rate
        super(LSTMModelModule, self).__init__()
        self.model = ACMTP10LSTMModel(input_size=input_size)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.correct_prediction_metric = CorrectPredictionPercentage(threshold=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.correct_prediction_metric.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        # Log the custom metric at the end of the test epoch
        self.log("test_percent_correct", self.correct_prediction_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR by 50% every 10 epochs
        return [optimizer], [scheduler]

# Define Loss Tracking Callback
class LossTracker(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log only every 10 epochs
        if (trainer.current_epoch + 1) % 10 == 0:
            train_loss = trainer.callback_metrics["train_loss"].item()
            self.train_losses.append(train_loss)
            print(f"Epoch {trainer.current_epoch + 1}: Training Loss = {train_loss:.6f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log only every 10 epochs
        if (trainer.current_epoch + 1) % 10 == 0:
            val_loss = trainer.callback_metrics["val_loss"].item()
            self.val_losses.append(val_loss)
            print(f"Epoch {trainer.current_epoch + 1}: Validation Loss = {val_loss:.6f}")

    def on_train_end(self, trainer, pl_module):
        # Display final table with only training and validation losses
        table_data = [
            ["Final Training Loss", f"{self.train_losses[-1]:.6f}"],
            ["Final Validation Loss", f"{self.val_losses[-1]:.6f}"]
        ]
        print("\nFinal Results:")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))

# Data Preparation (as per your original setup)
file_path = '/Users/fernandayepez/Documents/Neuro_Networks/ACMTermPremium.csv'
TP = pd.read_csv(file_path)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(TP[['ACMTP10']]), columns=['ACMTP10'])

# Function to create rolling sequences
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i: i + window_size])
    return np.array(sequences)

window_size = 5
X_seq = create_sequences(X.to_numpy(), window_size)
Y_seq = X['ACMTP10'][window_size:].to_numpy()

# Split data into train, validation, and test sets
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

# Convert data to TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

# Initialize model, loss tracker, and data loaders
input_size = X_seq.shape[2]
model_module = LSTMModelModule(input_size=input_size)
loss_tracker = LossTracker()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Early stopping callback to prevent overfitting
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)

# Trainer with Early Stopping and Loss Tracking callbacks
trainer = Trainer(
    deterministic=True,
    max_epochs=100,
    callbacks=[loss_tracker, early_stop_callback]
)

# Train and validate
trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test the model to calculate the percentage of predictions within a 10% range
trainer.test(model_module, dataloaders=test_loader)


# Plotting the training and validation losses over epochs (displayed every 10 epochs)
plt.figure(figsize=(10, 6))
plt.plot(range(10, 10 * len(loss_tracker.train_losses) + 1, 10), loss_tracker.train_losses, label="Training Loss")
plt.plot(range(10, 10 * len(loss_tracker.val_losses) + 1, 10), loss_tracker.val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.show()
