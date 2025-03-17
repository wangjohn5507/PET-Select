import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from src import args
import random
import os

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# -----------------------
# 1. Create synthetic dataset
# -----------------------
class CodeComplexityDataset(Dataset):
    """
    A simple dataset where:
    - Each sample has 5 features (complexity metrics).
    - Each label is a binary indicator (0 or 1).
    """
    def __init__(self, data_list, label_list):
        super().__init__()
        
        self.X = torch.tensor(data_list, dtype=torch.float32)
        
        self.y = torch.tensor(label_list, dtype=torch.float32)

        print(len(self.X))
        print(len(self.y))

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# 2. Define the model
# -----------------------
class CodeComplexityClassifier(nn.Module):
    """
    A simple feed-forward network for binary classification.
    Input size: 5
    Output size: 1 (binary classification, uses sigmoid activation)
    """
    def __init__(self):
        super(CodeComplexityClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)   # final output layer for binary classification
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)  # sigmoid output
        return x

# -----------------------
# 3. Training and evaluation
# -----------------------
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            # Move features and labels to GPU if available (optional)
            features = features
            labels = labels

            # Forward pass
            outputs = model(features)
            # Binary cross-entropy expects outputs of shape (batch, 1) and labels of shape (batch, 1)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            predicted = (outputs >= 0.5).float().view(-1)  # threshold at 0.5
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

def get_data_list_data_list(data):
    data_list = []
    label_list = []
    for per_data in data:
        features = []
        features.append(per_data['normalized_physical_loc'])
        features.append(per_data['normalized_cyclomatic_complexity'])
        features.append(per_data['normalized_halstead_complexity'])
        features.append(per_data['normalized_mi'])
        features.append(per_data['normalized_cognitive_complexity'])
        data_list.append(features)
        difficulty = per_data['meta_data']['difficulty']
        if difficulty == 'introductory':
            label_list.append(0)
        else:
            label_list.append(1)
        # label_list.append(per_data['label'])
    return data_list, label_list

def main(arguments):
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 16 

    # -----------------------
    # Prepare dataset
    # -----------------------
    train_path = f'5fold_dataset/{arguments.dataset}_{arguments.model}_train_{arguments.fold}.jsonl'
    test_path = f'5fold_dataset/{arguments.dataset}_{arguments.model}_test_{arguments.fold}.jsonl'
    train_data = list(map(json.loads, open(train_path)))
    test_data = list(map(json.loads, open(test_path)))
    
    train_data_list, train_label_list = get_data_list_data_list(train_data)
    test_data_list, test_label_list = get_data_list_data_list(test_data)
    
    train_dataset = CodeComplexityDataset(train_data_list, train_label_list)
    test_dataset = CodeComplexityDataset(test_data_list, test_label_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------
    # Initialize model, loss, optimizer
    # -----------------------
    model = CodeComplexityClassifier()

    num_pos = sum(train_label_list)
    num_neg = len(train_label_list) - num_pos
    print(num_pos, num_neg)
    pos_weight = torch.tensor([(num_neg/num_pos)+0.8], dtype=torch.float32)
    print(pos_weight)

    criterion = nn.BCELoss()  # binary cross-entropy loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # binary cross-entropy loss with class imbalance
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------
    # Train the model
    # -----------------------
    train_model(model, train_loader, criterion, optimizer, epochs=epochs)

    # -----------------------
    # Evaluate the model
    # -----------------------
    evaluate_model(model, test_loader)

    model_path = 'PET_model_result/complexity_model/complexity_model.pth'

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    set_seed(2)
    arguments = args.get_args()
    main(arguments)
