# make sure pytorch is installed before you run this code

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NeuralProbabilisticLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size, hidden_size):
        super(NeuralProbabilisticLanguageModel, self).__init__()
        
        # Word embeddings for each word in the vocabulary
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.hidden_layer = nn.Linear(context_size * embed_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, context_words):

        embeds = self.embeddings(context_words) 
        embeds = embeds.view(embeds.size(0), -1)
        hidden_output = torch.tanh(self.hidden_layer(embeds)) 
        output_scores = self.output_layer(hidden_output)
        probabilities = F.log_softmax(output_scores, dim=1) 
        
        return probabilities

vocab_size = 1000  
embed_size = 50    
context_size = 3   
hidden_size = 128 

model = NeuralProbabilisticLanguageModel(vocab_size, embed_size, context_size, hidden_size)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
context_words = torch.tensor([[12, 34, 56], [78, 90, 12]], dtype=torch.long)
target_words = torch.tensor([78, 34], dtype=torch.long)                

epochs = 5
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    log_probs = model(context_words)
    loss = loss_function(log_probs, target_words)
    loss.backward()

    optimizer.step()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

with torch.no_grad():
    model.eval()
    test_context = torch.tensor([[12, 34, 56]], dtype=torch.long)
    predictions = model(test_context)
    predicted_word = torch.argmax(predictions, dim=1)
    print("Predicted next word index:", predicted_word.item())