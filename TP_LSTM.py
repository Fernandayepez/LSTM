from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torchmetrics import Metric
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt
import random


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)  # Set the seed


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


# Define the Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, lstm_outputs):
        attn_output, attn_weights = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)
        context_vector = attn_output.mean(dim=1)  # Shape: (batch_size, hidden_size)
        return context_vector, attn_weights.mean(dim=1)  # Mean to reduce across heads


# Define the LSTM model with Attention
class ACMTP10LSTMWithEnhancedAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.3):
        super(ACMTP10LSTMWithEnhancedAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.dense = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_outputs, (h_n, c_n) = self.lstm(x)
        context_vector, attention_scores = self.attention(lstm_outputs)
        val = self.dense(self.dropout(context_vector))
        return torch.flatten(val), attention_scores


# LightningModule for training and testing with LSTM and Attention
class LSTMModelWithAttentionModule(LightningModule):
    def __init__(self, input_size, lr=0.0001):
        super(LSTMModelWithAttentionModule, self).__init__()
        self.model = ACMTP10LSTMWithEnhancedAttentionModel(input_size=input_size)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.correct_prediction_metric = CorrectPredictionPercentage(threshold=0.1)
        self.attention_scores_by_date = defaultdict(float)
        self.attention_scores_by_date_count = defaultdict(int)  # Count appearances of each date

    def forward(self, x):
        return self.model(x)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, attention_scores = self.model(x)
        mse_loss = self.loss_fn(preds, y)
        
        # Penalize over-concentration of attention
        attention_var_loss = torch.var(torch.mean(attention_scores, dim=1))
        total_loss = mse_loss + 0.1 * attention_var_loss  # Adjust weight as needed
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, attention_scores = self.model(x)
        loss = self.loss_fn(preds, y)
        self.correct_prediction_metric.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)

        # Log batch dates and attention scores
        start_idx = batch_idx * x.size(0)
        batch_dates = test_dates[start_idx: start_idx + x.size(0)]
        for i, scores in enumerate(attention_scores):
            for j, score in enumerate(scores):
                date = batch_dates[i][j]
                self.attention_scores_by_date[date] += score.item()
                self.attention_scores_by_date_count[date] += 1

        return loss

    def on_test_epoch_end(self):
        percent_correct = self.correct_prediction_metric.compute()
        self.log("test_percent_correct", percent_correct, prog_bar=True)

        normalized_scores = {
            date: score / self.attention_scores_by_date_count[date]
            for date, score in self.attention_scores_by_date.items()
        }

        top_dates = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop 20 Most Important Dates by Normalized Attention Score:")
        print(tabulate(top_dates, headers=["Date", "Normalized Attention Score"], tablefmt="grid"))

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]


# Data Preparation
file_path = '/Users/fernandayepez/Documents/Neuro_Networks/ACMTermPremium.csv'
data = pd.read_csv(file_path)
dates = data['DATE']  
ACMTP_data = data[['ACMTP10']]
scaler = StandardScaler()
ACMTP_data_scaled = pd.DataFrame(scaler.fit_transform(ACMTP_data), columns=['ACMTP10'])

def create_non_overlapping_sequences_with_dates(data, dates, window_size):
    sequences = []
    sequence_dates = []
    targets = []
    for i in range(0, len(data) - window_size, window_size):  # Step size = window_size
        sequences.append(data[i: i + window_size])
        sequence_dates.append(dates[i: i + window_size].to_numpy())
        targets.append(data[i + window_size])  # Add the target value
    return np.array(sequences), sequence_dates, np.array(targets)

def stratify_sequences_by_decade(X_seq, Y_seq, sequence_dates):
    # Flatten Y_seq if necessary
    Y_seq = np.array(Y_seq).flatten()  # Ensure Y_seq is 1-dimensional

    # Extract decades from sequence dates
    decades = [dates[0].split('-')[0] + "0s" for dates in sequence_dates]

    # Debugging: Check the shapes of inputs
    print(f"Y_seq shape after flatten: {Y_seq.shape}")
    print(f"Decades shape: {len(decades)}")
    print(f"Length of X_seq: {len(X_seq)}")

    # Ensure lengths match
    if len(Y_seq) != len(decades) or len(Y_seq) != len(X_seq):
        raise ValueError("Mismatch in lengths of X_seq, Y_seq, and decades.")

    # Create a DataFrame
    df = pd.DataFrame({"Y": Y_seq, "Decade": decades, "Index": range(len(X_seq))})

    # Downsample to equalize decade representation
    min_count = df['Decade'].value_counts().min()
    balanced_df = df.groupby('Decade').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

    # Use indices to extract balanced sequences and dates
    balanced_indices = balanced_df['Index'].tolist()
    X_balanced = X_seq[balanced_indices]
    Y_balanced = Y_seq[balanced_indices]
    dates_balanced = [sequence_dates[i] for i in balanced_indices]
    return X_balanced, Y_balanced, dates_balanced


window_size = 5
X_seq, sequence_dates, Y_seq = create_non_overlapping_sequences_with_dates(
    ACMTP_data_scaled.to_numpy(), dates, window_size
)
X_seq_balanced, Y_seq_balanced, sequence_dates_balanced = stratify_sequences_by_decade(X_seq, Y_seq, sequence_dates)

X_train_val, X_test, Y_train_val, Y_test, train_val_dates, test_dates = train_test_split(
    X_seq_balanced, Y_seq_balanced, sequence_dates_balanced, test_size=0.2, random_state=42
)
X_train, X_val, Y_train, Y_val, train_dates, val_dates = train_test_split(
    X_train_val, Y_train_val, train_val_dates, test_size=0.25, random_state=42
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

input_size = X_seq.shape[2]
model_module = LSTMModelWithAttentionModule(input_size=input_size)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)

trainer = Trainer(deterministic=True, max_epochs=100, callbacks=[early_stop_callback])

trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model_module, dataloaders=test_loader)

def compute_scores_by_decade(attention_scores_by_date, attention_scores_by_date_count):
    # Calculate cumulative scores per decade
    cumulative_scores = defaultdict(float)
    normalized_scores = defaultdict(float)

    for date, score in attention_scores_by_date.items():
        decade = date.split('-')[0] + "0s"
        cumulative_scores[decade] += score
        normalized_scores[decade] += score / attention_scores_by_date_count[date]

    return cumulative_scores, normalized_scores


def plot_scores_by_decade(cumulative_scores, normalized_scores):
    # Sort decades for consistent plotting
    decades = sorted(cumulative_scores.keys())
    cumulative_values = [cumulative_scores[decade] for decade in decades]
    normalized_values = [normalized_scores[decade] for decade in decades]

    # Plot cumulative attention scores
    plt.figure(figsize=(10, 6))
    plt.bar(decades, cumulative_values, alpha=0.7)
    plt.xlabel("Decade")
    plt.ylabel("Cumulative Attention Score")
    plt.title("Cumulative Attention Scores by Decade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot normalized attention scores
    plt.figure(figsize=(10, 6))
    plt.bar(decades, normalized_values, alpha=0.7, color="orange")
    plt.xlabel("Decade")
    plt.ylabel("Normalized Attention Score")
    plt.title("Normalized Attention Scores by Decade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Compute scores
cumulative_scores, normalized_scores = compute_scores_by_decade(
    model_module.attention_scores_by_date, model_module.attention_scores_by_date_count
)

# Plot scores
plot_scores_by_decade(cumulative_scores, normalized_scores)

# Calculate variance of attention scores by decade
def calculate_attention_score_variance(attention_scores_by_date):
    decade_scores = defaultdict(list)
    for date, score in attention_scores_by_date.items():
        decade = date.split('-')[-1][:3] + "0s"
        decade_scores[decade].append(score)
    
    # Compute variance for each decade
    decade_variance = {decade: np.var(scores) for decade, scores in decade_scores.items()}
    return decade_variance

variance_by_decade = calculate_attention_score_variance(model_module.attention_scores_by_date)
print("\nVariance of Attention Scores by Decade:")
for decade, var in sorted(variance_by_decade.items()):
    print(f"{decade}: {var}")

# Calculate total attention score contribution of top 20 dates
def top_scores_contribution(top_dates, attention_scores_by_date):
    total_score = sum(attention_scores_by_date.values())  # Total attention score across all dates
    top_score = sum(attention_scores_by_date[date] for date in top_dates)  # Total score for top 20 dates
    return top_score / total_score


# Extract the top 20 dates# Extract the top 20 dates (only the date, not the scores)
top_20_dates = [date for date, _ in sorted(model_module.attention_scores_by_date.items(), key=lambda x: x[1], reverse=True)[:20]]

# Calculate the contribution
contribution = top_scores_contribution(top_20_dates, model_module.attention_scores_by_date)
print(f"\nContribution of Top 20 Dates to Total Attention Score: {contribution:.2%}")


# Add regularization to penalize over-concentration on specific dates
def training_step(self, batch, batch_idx):
    x, y = batch
    preds, attention_scores = self.model(x)
    mse_loss = self.loss_fn(preds, y)
    
    # Penalize high variance in attention scores across sequences
    attention_var_loss = torch.var(torch.mean(attention_scores, dim=1))
    total_loss = mse_loss + 0.1 * attention_var_loss  # Adjust weight as needed
    
    self.log("train_loss", total_loss, prog_bar=True)
    return total_loss
# Visualize attention scores for top 20 sequences
def plot_top_sequences_attention(top_dates, attention_scores_by_date):
    top_scores = [attention_scores_by_date[date] for date, _ in top_dates]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_scores)), top_scores)
    plt.xlabel("Top 20 Sequences")
    plt.ylabel("Attention Scores")
    plt.title("Attention Scores for Top 20 Sequences")
    plt.show()

plot_top_sequences_attention(top_20_dates, model_module.attention_scores_by_date)

# Log-transform attention scores
def log_normalize_attention_scores(attention_scores_by_date):
    min_score = min(attention_scores_by_date.values())
    log_scores = {date: np.log(score - min_score + 1) for date, score in attention_scores_by_date.items()}
    return log_scores

# Apply log transformation
log_attention_scores = log_normalize_attention_scores(model_module.attention_scores_by_date)

# Update the top 20 list based on log-transformed scores
top_20_dates_log = [date for date, _ in sorted(log_attention_scores.items(), key=lambda x: x[1], reverse=True)[:20]]
print("\nTop 20 Dates After Log Normalization:")
for date in top_20_dates_log:
    print(date)

