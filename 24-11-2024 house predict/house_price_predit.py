import torch
import os
import pandas as pd
from torch.utils.data import TensorDataset

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


data_dir = os.path.join(os.getcwd(), 'house-prices-advanced-regression-techniques')

train_data = pd.read_csv(data_dir + '/train.csv')
test_data = pd.read_csv(data_dir + '/test.csv')

# Save test IDs before preprocessing
test_ids = test_data['Id'].values
# Feature Engineering
def preprocess(train_data, test_data):

    target = train_data['SalePrice'].values.reshape(-1, 1)  # Reshape for scaler

    train_data = train_data.drop(['Id', 'SalePrice'], axis=1)
    test_data = test_data.drop('Id', axis=1)

    label_encoder = LabelEncoder()
    common_columns = train_data.columns.intersection(test_data.columns)

    for column in common_columns:

        if train_data[column].dtype in  ['int64','float64']:
            print(f'column checking is: {column}')
            train_data[column] = train_data[column].fillna(train_data[column].mean())
            test_data[column] = test_data[column].fillna(test_data[column].mean())

        else:
            train_data[column] = train_data[column].fillna(train_data[column].mode())
            test_data[column] = test_data[column].fillna(test_data[column].mode())


            if train_data[column].dtype in ['object']:
                all_data =  pd.concat([train_data[column], test_data[column]])
                label_encoder.fit(all_data.astype(str))

                train_data[column] = label_encoder.transform(train_data[column].astype(str))
                test_data[column] = label_encoder.transform(test_data[column].astype(str))

    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    # Scale target (need to reshape for 1D array)
    target_scaler = StandardScaler()
    target = target_scaler.fit_transform(target).flatten()

    return train_data, test_data, target, target_scaler

train_data, test_data, target, target_scaler = preprocess(train_data, test_data)

# Calculate split index (20% for validation)
train_size = 0.8

# Use train_test_split for proper random splitting
X_train, X_val, y_train, y_val = train_test_split(
    train_data,
    target,
    train_size=train_size,
    random_state=24  # for reproducibility
)


# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train.values).float()
X_val = torch.from_numpy(X_val.values).float()
y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()

y_train = y_train.view(-1, 1)  # Reshape to [batch_size, 1]
y_val = y_val.view(-1, 1)      # Reshape to [batch_size, 1]
# Print shapes to verify
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

# Create datasets after verifying shapes
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


class ANN(nn.Module):
    def __init__(self, in_channel, hidden_dim, output_dim, dropout_rate=0.3):
        super(ANN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.layer(x)


in_channel = X_train.shape[1]
model = ANN(in_channel=in_channel, hidden_dim=128, output_dim=1, dropout_rate=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

def train(model, train_loader, val_loader, num_epochs, optimizer):

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epoch_without_improvement = 0
    patience = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(train_loader)
        val_losses.append(epoch_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {epoch_train_loss:.4f}')
            print(f'Validation Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epoch_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epoch_without_improvement += 1

        if epoch_without_improvement >= patience:
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break

    return train_losses, val_losses

train_loss, val_loss = train(model, train_loader, val_loader, 1000, optimizer)


# Update the prediction function to ensure 1-dimensional output
def predict(model, test_data, target_scaler):
    model.eval()
    # Convert DataFrame to tensor
    test_tensor = torch.from_numpy(test_data.values).float()

    with torch.no_grad():
        predictions = model(test_tensor)
        # Convert to numpy and ensure it's 2D for inverse transform
        predictions = predictions.numpy()
        # Inverse transform
        predictions = target_scaler.inverse_transform(predictions)
        # Flatten to 1D array
        predictions = predictions.flatten()

    # Print shape for debugging
    print("Predictions shape after processing:", predictions.shape)
    return predictions


# Load best model with weights_only=True to address the warning
model.load_state_dict(torch.load('best_model.pt', weights_only=True))

# Make predictions
pred = predict(model, test_data, target_scaler)

# Create submission DataFrame with flattened predictions
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': pred
})
submission.to_csv('house_price_predict.csv', index=False)


# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()