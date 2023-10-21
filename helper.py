import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from SimpleNN import SimpleNN
import time

# Converts categorical columns to one-enocding columns, drops rows with NA values, converts bools to ints
def clean_dataframe(df):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Drop rows with NA values
    df.dropna(inplace=True)
    
    # Convert boolean columns to int
    bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df


#'train.csv'
def preprocess_titanic_dataset(filename):
    print('Preprocessing')
    # Read data from CSV file
    df = pd.read_csv(filename)  # Replace 'your_data.csv' with your actual file path
    # print("Before:")
    # print(df.head)
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df = clean_dataframe(df)
    # print("After:")
    # print(df.head)

    for col in ['Pclass', 'Age', 'Fare']:
        df[f'{col}_squared'] = df[col] ** 2

    # # Extract features and labels
    X = df[['Pclass', 'Sex_female', 'Sex_male', 'Age_squared']].values
    y = df['Survived'].values
    return X, y

def split_training_and_validation_sets(X, y):
    # # Convert to PyTorch tensors
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

def convert_to_tensors(X, y, device=None):
    if device is not None:
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    else:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return X, y

def train_neural_net(model, n_epochs, X_train, y_train, X_val, y_val, filename):

    X_train, y_train = convert_to_tensors(X_train, y_train, model.device)
    X_val, y_val = convert_to_tensors(X_val, y_val, model.device)

    # Initialize model, loss, and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    accuracy_train_list = []
    accuracy_val_list = []


    start_time = time.time()
    for epoch in range(n_epochs):
        # Forward pass on training data
        outputs_train = model(X_train)
        loss_train = criterion(outputs_train, y_train)
        accuracy_train = accuracy(outputs_train, y_train)
        accuracy_train_list.append(accuracy_train)
        
        # Forward pass on validation data
        with torch.no_grad():
            outputs_val = model(X_val)
            loss_val = criterion(outputs_val, y_val)
            accuracy_val = accuracy(outputs_val, y_val)
            accuracy_val_list.append(accuracy_val)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # Print epoch status
        if (epoch + 1) % ((n_epochs) / 10) == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}]')

        # if (epoch+1) % 50 == 0:
        #     print(f'Epoch [{epoch+1}/{n_epochs}], Loss (Train): {loss_train.item():.4f}, Accuracy (Train): {accuracy_train:.4f}, Loss (Val): {loss_val.item():.4f}, Accuracy (Val): {accuracy_val:.4f}')

    end_time = time.time()
    print(f"Training complete. Took {end_time - start_time:.2f}s")
    torch.save(model.state_dict(), filename)
    return accuracy_train_list, accuracy_val_list

def accuracy(y_pred, y):
    # Assuming y_pred are the model predictions and y_true are the true labels, both as PyTorch tensors
    y_pred_rounded = torch.round(y_pred)
    correct = (y_pred_rounded == y).float().sum()
    accuracy = correct / y.shape[0]
    return accuracy
