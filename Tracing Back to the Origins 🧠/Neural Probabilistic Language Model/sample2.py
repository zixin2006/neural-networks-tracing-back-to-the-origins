import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
import random

from NPLM import NeuralProbabilisticLanguageModel

# a small sample text corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat lay on the mat",
    "the dog lay on the log",
    "a cat sat on a mat"
]

# Build Vocabulary
def build_vocab(corpus):
    word2idx = {}
    idx2word = {}
    idx = 0
    for sentence in corpus:
        for word in sentence.split():
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    return word2idx, idx2word

word2idx, idx2word = build_vocab(corpus)
vocab_size = len(word2idx)

# Context-Target Pairs
def generate_data(corpus, word2idx, context_size=3):
    data = []
    for sentence in corpus:
        words = sentence.split()
        for i in range(context_size, len(words)):
            context = [word2idx[words[i - j - 1]] for j in range(context_size)]
            target = word2idx[words[i]]
            data.append((context, target))
    return data

context_size = 3  
data = generate_data(corpus, word2idx, context_size)

embed_size = 10
hidden_size = 20
epochs = 100

model = NeuralProbabilisticLanguageModel(vocab_size, embed_size, context_size, hidden_size)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
context_tensors = torch.tensor([x[0] for x in data], dtype=torch.long)
target_tensors = torch.tensor([x[1] for x in data], dtype=torch.long)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    log_probs = model(context_tensors)
    loss = loss_function(log_probs, target_tensors)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# Test the model
test_context = [word2idx["the"], word2idx["cat"], word2idx["sat"]]
test_context_tensor = torch.tensor([test_context], dtype=torch.long)

with torch.no_grad():
    model.eval()
    predictions = model(test_context_tensor)
    predicted_idx = torch.argmax(predictions, dim=1).item()
    print("Predicted next word:", idx2word[predicted_idx])