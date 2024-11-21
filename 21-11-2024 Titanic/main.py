import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data_dir = os.path.join(os.getcwd(), 'titanic')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

def proprocess(train,test):
    train = train.copy()
    test = test.copy()

    columns_to_drop = ['Name','Ticket', 'Cabin']
    train = train.drop(columns_to_drop, axis=1)
    test = test.drop(columns_to_drop, axis=1)

    label_encoder = LabelEncoder()
    categorical_columns = ['Sex', 'Embarked']

    for column in categorical_columns:
        # Fill NA values before encoding
        train[column] = train[column].fillna(train[column].mode()[0])
        test[column] = test[column].fillna(test[column].mode()[0])

        # Encode categorical variables
        combined_data = pd.concat([train[column], test[column]])
        label_encoder.fit(combined_data.astype(str))
        train[column] = label_encoder.transform(train[column].astype(str))
        test[column] = label_encoder.transform(test[column].astype(str))

    numerical_column = ['Age', 'Fare']
    for column in numerical_column:
        train[column] = train[column].fillna(train[column].mean())
        test[column] = test[column].fillna(test[column].mean())

    scaler = StandardScaler()
    data_columns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    train[data_columns] = scaler.fit_transform(train[data_columns])
    test[data_columns] = scaler.transform(test[data_columns])
    return train, test

train_process, test_process = proprocess(train,test)

# Prepare training data
X_train = torch.tensor(train_process.drop(['Survived', 'PassengerId'], axis=1).values, dtype=torch.float32)

y_train = train['Survived'].values
y_train = torch.tensor(y_train, dtype=torch.long)  # Use the saved target variable

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# classification problem

class ANN(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(in_channel, hidden_dim, bias=True)
        self.layer2 = nn.Linear(hidden_dim, 2, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        output = self.relu((self.layer1(x)))
        output = self.dropout(output)
        output = self.layer2(output)
        return output

model = ANN(X_train.shape[1], 128)
losses = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, train_dataset, epochs, optimizer):
    model.train()
    train_loss = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_dataset:

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = losses(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss /len(train_dataset)
        train_loss.append(epoch_loss)

        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

    return train_loss

def test(model, test_dataset):
    model.eval()
    total_loss = 0.0
    num_loop = 0
    for input, labels in test_dataset:
        with torch.no_grad():
            pred = model(input)
            loss = losses(pred - labels)
            total_loss += loss.item()
            num_loop += 1
            print(f' error is :{total_loss / num_loop}')
    return total_loss/num_loop

train_loss = train(model, train_dataset, 1000, optimizer)

plt.figure(figsize=(10, 8))
plt.plot(train_loss, label='train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
test_dataset = torch.tensor(test_process.drop(['PassengerId'], axis=1).values,dtype=torch.float32)
with torch.no_grad():
    pred = model(test_dataset)
    predicted = torch.argmax(pred, dim=1)

submission = pd.DataFrame({'PassengerId': test_process['PassengerId'], 'Survived': predicted})
submission.to_csv('submission.csv', index=False)
