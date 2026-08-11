import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


print("Loading data...")
df = pd.read_csv("data_set.csv")

X = df[['d1', 'd2', 'd3', 'angle1', 'angle2']].values
y = df[['T1', 'T2', 'T3']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inputs 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)), batch_size=128, shuffle=True)


class SurrogateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus() # Forces all predicted times to be > 0 mathematically
        )
        
    def forward(self, x):
        return self.net(x)

# training
model = SurrogateMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 200
print(f"Training MLP on {len(X_train)} perfect samples...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss (MSE): {running_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(torch.FloatTensor(X_test_scaled))
    mae = torch.mean(torch.abs(test_preds - torch.FloatTensor(y_test)))
    
print(f"\nFinal Test Mean Absolute Error: {mae.item():.4f} seconds")

dummy_input = torch.randn(1, 5)
torch.onnx.export(model, dummy_input, "surrogate_model.onnx", 
                  input_names=['geometry_vector'], output_names=['target_times'])
print("Saved ONNX model for geometric controller integration.")