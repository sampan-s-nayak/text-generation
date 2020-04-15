import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_loader import dataloader as dl
from Settings import settings as default

# define our neural network
class TextGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) # embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.next_word = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x,prev_state):
        embeds = self.word_embeddings(x)
        lstm_out, state = self.lstm(embeds,prev_state)
        next_word = self.next_word(lstm_out)
        return next_word,state
