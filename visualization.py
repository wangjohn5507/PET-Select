from sentence_transformers import SentenceTransformer
import json

dataset = 'code_complex'
model_path = f'PET_model_result/{dataset}_contrastive_model'


# Load your trained sentence-transformer model or a pre-trained one
model = SentenceTransformer(model_path)
# model = SentenceTransformer('microsoft/codebert-base')

data = list(map(json.loads, open('5fold_dataset/APPS_gpt-4o_train_0.jsonl')))
sentences = []
labels = []
for per_data in data:
    sentence = per_data['prompt']
    complexity = per_data['code_complexity']
    if complexity * 100 < 20:
        label = 0
    else:
        label = 1
    sentences.append(sentence)
    labels.append(label)

print(len(sentences))
print(len(labels))

# exit()

# Example sentences to embed
# sentences = ["This is an example sentence.", "This is another sentence for comparison."]
# sentence1 = 'Write a function to get the word with most number of occurrences in the given strings list.'
# sentence2 = 'Write a python function to remove even numbers from a given list.'
# sentence3 = 'Write a function to find the maximum product subarray of the given array.'

# Generate embeddings
embeddings = model.encode(sentences)

# embedding_a = model.encode(sentence1)
# embedding_b = model.encode(sentence2)
# embedding_c = model.encode(sentence3)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# # Reduce dimensionality using PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# # # Create a scatter plot of the 2D embeddings
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], s=100)

# # Customize plot
# plt.title("PCA Visualization of Sentence Embeddings")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.show()

import numpy as np

# Example sentence embeddings
# embedding_a = np.array(embedding_a)
# embedding_b = np.array(embedding_b)
# embedding_c = np.array(embedding_c)

# Cosine similarity
# cosine_similarity = np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
# cosine_similarity2 = np.dot(embedding_a, embedding_c) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_c))

# print(1-cosine_similarity, 1-cosine_similarity2)




plt.figure(figsize=(10, 7))
print(len(embeddings_2d[:, 0]))
print(len(embeddings_2d[:, 1]))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette=["blue", "orange"], s=100)

plt.title("PCA Visualization of Contrastive Learning Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()
