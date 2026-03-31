import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last hidden state from the sequence
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(df_features, df_target, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(df_features) - seq_length):
        X_seq.append(df_features.iloc[i:i+seq_length].values)
        y_seq.append(df_target.iloc[i+seq_length])
        
    return np.array(X_seq), np.array(y_seq)
    
def train_lstm_model(df, seq_length=10, epochs=20, batch_size=32):
    print("\n" + "="*50)
    print("🧠 TRAINING LSTM TIME-SERIES MODEL")
    print("="*50)
    
    # Preprocessing
    exclude = ['Open', 'High', 'Low', 'Close', 'Target_Return', 'Target_Class', 'Risk_Level']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features].ffill().fillna(0)
    y = df['Target_Class']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Create Sequences
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
    
    # Time-Series Split (Walk-forward without leakage)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_data = TensorDataset(X_train_t, y_train_t)
    # Important: NO SHUFFLE for time series temporal integrity if using stateful LSTM!
    # But for stateless LSTM with windows, shuffling train loader is standard actually. 
    # To prevent look-ahead bias, test set evaluation must be strictly post-training in time. 
    # Since we sliced temporal arrays above (split_idx), test set is only the future. We can shuffle train batches safely.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
    
    # Model Setup
    model = StockLSTM(input_size=len(features), hidden_size=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Batch Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_binary = (test_preds > 0.5).float()
        acc = (test_preds_binary == y_test_t).float().mean()
        print(f"✅ LSTM Test Accuracy (Out-Of-Sample): {acc.item():.4f}")
        
    return model, scaler, features

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.dataset_builder import prepare_dataset

    df = prepare_dataset("RELIANCE.NS", "2018-01-01", "2024-01-01", horizon=1)
    if df is not None:
        model, scaler, feature_cols = train_lstm_model(df)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'scaler': scaler,
            'features': feature_cols
        }, "models/lstm_model.pth")
        print("\n💾 LSTM Model saved to models/lstm_model.pth")
