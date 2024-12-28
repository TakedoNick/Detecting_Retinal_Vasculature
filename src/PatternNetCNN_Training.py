import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_path = r'..\..\train_data_paper\normalizedData_2_sumzer.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

x_array = torch.tensor(data['Nor_train_NN'], dtype=torch.float32).T  # Transpose to match PyTorch input shape
v1 = torch.tensor(data['comp_target_NN'], dtype=torch.float32)

# Get input and output dimensions
R, Q = x_array.shape
S2 = (R * 2) + 1
output_dim = v1.shape[0]

# Define the PatternNet model
class PatternNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PatternNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = PatternNet(input_dim=R, hidden_dim=S2, output_dim=output_dim).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


x_train, x_val, y_train, y_val = train_test_split(x_array, v1.T, test_size=0.2, random_state=42)
x_train, x_val = x_train.cuda(), x_val.cuda()
y_train, y_val = y_train.cuda(), y_val.cuda()

# Training
epochs = 5000
best_val_loss = float('inf')
patience = int(Q / 100)
early_stopping_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_pattern_net.pth')  # Save the best model
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Print loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Evaluate the model
model.load_state_dict(torch.load('best_pattern_net.pth'))  # Load the best model
model.eval()
with torch.no_grad():
    train_outputs = model(x_array.cuda())
    train_predictions = (train_outputs > 0.5).cpu().numpy()
    train_labels = v1.cpu().numpy()
    train_acc = accuracy_score(train_labels, train_predictions)

print(f"Training Accuracy: {train_acc:.4f}")

# Save the model
torch.save(model.state_dict(), 'PatNet_CNN.pth')