import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import random
import tqdm
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2)

# Customize Dataset
class CustomDataset(Dataset):
    def __init__(self, embeddings, labels, ranks):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ranks = ranks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.ranks[idx]
    
from sentence_transformers import SentenceTransformer
def get_embedding(questions, model_path):
    model = SentenceTransformer(model_path)
    embeddings = []
    print('Generating embeddings...')
    for question in tqdm.tqdm(questions, ncols=75):
        embedding = model.encode(question)
        embeddings.append(embedding)
    return embeddings

class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = f'PET_model_dataset/code_complex_classification_dataset_train.jsonl'
test_file_path = f'PET_model_dataset/code_complex_classification_dataset_test.jsonl'

model_path = f'PET_model_result/code_complex_contrastive_model'
# model_path = 'microsoft/codebert-base'
data = pd.read_json(file_path, lines=True)
test_data = pd.read_json(test_file_path, lines=True)
data = list(map(json.loads, open(file_path)))
test_data = list(map(json.loads, open(test_file_path)))

label_dict = {'Zeroshot': 0, 'Zeroshot_CoT': 1, 'Fewshot': 2, 'Fewshot_CoT': 3, 'Persona': 4, 'Self-planning': 5, 'Self-refine': 6, 'Progressive-Hint': 7, 'Self-debug': 8}

def generate_list(data):
    questions, labels, ranks = [], [], []
    for per_data in data:
        rank = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}
        question = per_data['prompt']
        label = label_dict[per_data['ranked_techniques'][0][0]]
        for idx, ranked_technique in enumerate(per_data['ranked_techniques']):
            ranked_label = label_dict[ranked_technique[0]]
            rank[str(idx)] = ranked_label
        questions.append(question)
        labels.append(label)
        ranks.append(rank)
    return questions, labels, ranks
    

questions, labels, ranks = generate_list(data)
test_questions, test_labels, test_ranks = generate_list(test_data)

# embeddings = data['embedding'].tolist()
# test_embeddings = test_data['embedding'].tolist()

embeddings = get_embedding(questions, model_path)
test_embeddings = get_embedding(test_questions, model_path)

def dcg(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg(relevances, k):
    dcg_value = dcg(relevances, k)
    idcg_value = dcg(sorted(relevances, reverse=True), k)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value

def custom_collate_fn(batch):
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    relevance_scores = [item[2] for item in batch]
    return embeddings, labels, relevance_scores

# hyperparameter setting
input_size = len(embeddings[0])
print(input_size)
num_classes = 9
print(num_classes)
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Split train, eval, and test data
# train_embeddings, temp_embeddings, train_labels, temp_labels, train_ranks, temp_ranks = train_test_split(embeddings, labels, ranks, test_size=0.3, random_state=42)
# eval_embeddings, test_embeddings, eval_labels, test_labels, eval_ranks, test_ranks = train_test_split(temp_embeddings, temp_labels, temp_ranks, test_size=0.5, random_state=42)

train_embeddings, eval_embeddings, train_labels, eval_labels, train_ranks, eval_ranks = train_test_split(embeddings, labels, ranks, test_size=0.2, random_state=42)
test_embeddings, test_labels, test_ranks = test_embeddings, test_labels, test_ranks


# create CustomDataset
train_dataset = CustomDataset(train_embeddings, train_labels, train_ranks)
eval_dataset = CustomDataset(eval_embeddings, eval_labels, eval_ranks)
test_dataset = CustomDataset(test_embeddings, test_labels, test_ranks)

# create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = [0.5,  0.5,  0.5,  0.5,  1, 0.5]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
# print(class_weights)

# create model, criterion, and optimizer
model = ClassificationModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# calcuate MRR
def calculate_mrr(outputs, labels):
    mrr = 0.0
    for i in range(outputs.size(0)):
        scores = outputs[i]
        target = labels[i].item()
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank
    return mrr / outputs.size(0)

#calculate ndcg
def calculate_ndcg(outputs, ranks, k=6):
    ndcg_value = 0.0
    for i in range(outputs.size(0)):
        rank = ranks[i]
        scores = outputs[i]
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        relevances = np.array([rank[str(idx.item())] for idx in sorted_indices])
        ndcg_value += ndcg(relevances, k)
    return ndcg_value / outputs.size(0)

# train_model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for embeddings_batch, labels_batch, ranks_batch in train_dataloader:
        # print(ranks_batch)
        embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = 100 * correct / total


    # evaluate model
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_mrr = 0.0
    val_ndcg = 0.0
    val_ndcg_total = 0
    with torch.no_grad():
        for val_embeddings_batch, val_labels_batch, val_ranks_batch in eval_dataloader:
            val_embeddings_batch, val_labels_batch = val_embeddings_batch.to(device), val_labels_batch.to(device)

            val_outputs = model(val_embeddings_batch)
            val_loss = criterion(val_outputs, val_labels_batch)
            val_running_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels_batch.size(0)
            val_ndcg_total += val_outputs.size(0)
            val_correct += (val_predicted == val_labels_batch).sum().item()

            val_mrr += calculate_mrr(val_outputs, val_labels_batch) * val_labels_batch.size(0)
            val_ndcg += calculate_ndcg(val_outputs, val_ranks_batch, num_classes) * val_outputs.size(0)

    val_epoch_loss = val_running_loss / len(eval_dataloader)
    val_epoch_acc = 100 * val_correct / val_total
    val_epoch_mrr = val_mrr / val_total
    val_epoch_ndcg = val_ndcg / val_ndcg_total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%, Val MRR: {val_epoch_mrr:.4f}, Val nDCG: {val_epoch_ndcg:.4f}')


def evaluate_nDCG(model, dataloader, device, k=6):
    model.eval()
    ndcg_total = 0.0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, rank_batch in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)

            for i in range(outputs.size(0)):
                rank = rank_batch[i]
                # print(rank)
                scores = outputs[i]
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)

                # first_two_choices = [0, 4]
                # remaining_choices = [1, 2, 3, 5]
                # # Generate the list
                # random_labels = random.sample(first_two_choices, 2) + random.sample(remaining_choices, 4)
                # # print(target)
                # sorted_indices = torch.tensor(random_labels)
                
                relevances = np.array([rank[str(idx.item())] for idx in sorted_indices])
                # print(relevances)
                ndcg_total += ndcg(relevances, k)
                total += 1

    ndcg_avg = ndcg_total / total
    print(f'nDCG@{k}: {ndcg_avg:.4f}')

def evaluate_mrr(model, dataloader, device):
    model.eval()
    mrr = 0.0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)
            for i in range(outputs.size(0)):
                scores = outputs[i]
                # print(scores)
                target = labels_batch[i].item()
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                print(target, sorted_indices)

                # # Define the lists for random choices
                # first_two_choices = [0, 4]
                # remaining_choices = [1, 2, 3, 5]

                # # Generate the list
                # random_labels = random.sample(first_two_choices, 2) + random.sample(remaining_choices, 4)
                # # print(target)
                # sorted_indices = torch.tensor(random_labels)

                rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
                mrr += 1.0 / rank
                total += 1

    mrr = mrr / total
    print(f'MRR: {mrr:.4f}')

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in dataloader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)

            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs.data, 1)

            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')


evaluate_nDCG(model, test_dataloader, device, num_classes)
evaluate_mrr(model, test_dataloader, device)
evaluate_model(model, test_dataloader, device)

output_model_path = f'PET_model_result/classification_model/code_complex_classification_model_parameters.pth'
torch.save(model.state_dict(), output_model_path)