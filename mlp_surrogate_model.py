import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

df = pd.read_csv("lag_aware_trajectory_data.csv")
X = df[['d', 'theta', 'phi', 'v0', 'vf']].values
y = df[['Target_T']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# PyTorch Tensors
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True)

# MLP architecture
class LagAwareMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
    def forward(self, x):
        return self.network(x)

model = LagAwareMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# training loop
print("Training MLP Surrogate Model...")
epochs = 100
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

print("Training Complete. Exporting to ONNX...")

# MATLAB integration
dummy_input = torch.randn(1, 5) # 5 features
torch.onnx.export(model, dummy_input, "lag_aware_mlp.onnx", 
                  input_names=['state_vector'], output_names=['target_T'])