"""this code is taken from my liquidity project in my other repository"""


from sklearn.preprocessing import StandardScaler
from data_features import feature_table, tickers
import numpy 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seq_len = 20 

class GRUDataset(Dataset):
    def __init__(self, features, target, seq_len):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, i):
        x = self.X[i : i + self.seq_len]
        y = self.y[i + self.seq_len - 1]
        return x, y


table = feature_table["GLD"].dropna()
feature_cols = [c for c in table.columns if c not in ("amihud", "cs_spread")]

X = table[feature_cols].to_numpy()     
y = table["amihud"].to_numpy()         

n = len(table)
test_start = int(n * 0.8)
val_start  = int(test_start * 0.85)

X_train, y_train = X[:val_start],           y[:val_start]
X_val,   y_val   = X[val_start:test_start], y[val_start:test_start]
X_test,  y_test  = X[test_start:],          y[test_start:]

# 3. Scale using TRAIN statistics only
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

X_train, X_val, X_test = (x_scaler.transform(a) for a in (X_train, X_val, X_test))
y_train, y_val, y_test = (y_scaler.transform(a.reshape(-1, 1)).ravel()
                          for a in (y_train, y_val, y_test))

train_ds = GRUDataset(X_train, y_train, seq_len)
val_ds   = GRUDataset(X_val,   y_val,   seq_len)
test_ds  = GRUDataset(X_test,  y_test,  seq_len)

# 5. One DataLoader per split
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)


class TimeSeriesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, h_final = self.gru(x)
        last = outputs[:, -1]
        return self.out(last)

xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)   # expect (32, 20, n_features) and (32, 1)


import copy

torch.manual_seed(0)
model = TimeSeriesGRU(input_size=xb.shape[2], hidden_size=16
                      , num_layers=1, output_size=1).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val, best_state, patience, bad_epochs = float("inf"), None, 10, 0

for epoch in range(200):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += loss_fn(model(xb), yb).item() * len(xb)
    val_loss /= len(val_ds)

    print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

    if val_loss < best_val:
        best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break

model.load_state_dict(best_state)   



from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model.eval()
preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
preds = np.vstack(preds)                                   # (n_windows, 1), standardized

pred_real = y_scaler.inverse_transform(preds).ravel()      # back to real Amihud units
true_real = y_scaler.inverse_transform(
    y_test[seq_len - 1:].reshape(-1, 1)).ravel()           # matching true values

rmse = np.sqrt(mean_squared_error(true_real, pred_real))
r2 = r2_score(true_real, pred_real)

# naive baseline: yesterday's value
persist = np.sqrt(mean_squared_error(true_real[1:], true_real[:-1]))
print(f"GRU    RMSE {rmse:.3e}  R2 {r2:.3f}")
print(f"Naive  RMSE {persist:.3e}")

