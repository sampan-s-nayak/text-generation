import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from Data_loader import dataloader as dl
from Settings import settings as default

corpus = dl.TextCorpus()

# define our neural network
class TextGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) # embedding layer (learns the embedding matrix)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,batch_first=True)
        self.next_word = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x,prev_state):
        embeds = self.word_embeddings(x)
        lstm_out, state = self.lstm(embeds,prev_state)
        next_word = self.next_word(lstm_out)
        return next_word,state

    def zero_state(self, batch_size):
        #returns initial LSTM stage
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

def get_loss_and_optimizer(model,learning_rate=default.LEARNING_RATE):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def train(model,epoch):
    loss_fn,optimizer = get_loss_and_optimizer(model)
    dataloader = DataLoader(corpus, batch_size=default.BATCH_SIZE, num_workers=4,shuffle=False)
    for i in range(epoch):
        model.train()
        epoch_loss = 0
        for j,data in enumerate(dataloader):
            # set gradients to zero
            optimizer.zero_grad()

            # initial LSTM stage
            state_h, state_c = model.zero_state(default.BATCH_SIZE)

            pred, (state_h, state_c) = model(data["curr_word"], (state_h, state_c))

            # calculate loss
            loss = loss_fn(pred.transpose(1, 2), data["next_word"])

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            # perform gradient clipping to prevent gradients from exploding
            _ = torch.nn.utils.clip_grad_norm_(
                model.parameters(), default.GRADIENTS_NORM)

            # Update the network's parameters
            optimizer.step()

            epoch_loss += loss_value
        print(f"epoch: {i+1} loss: {loss}")

def save_model(model,path=default.WEIGHTS_PATH):
    torch.save(model.state_dict(), path)

def load_model(path=default.WEIGHTS_PATH):
    model = TextGenerator(default.EMBEDDING_DIM,default.HIDDEN_LAYER_DIM,default.VOCAB_SIZE)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def generate_text(model,initial_text,length):
    # initial LSTM stage
    state_h, state_c = model.zero_state(default.BATCH_SIZE)
    print(initial_text,end=" ")
    pred, (state_h, state_c) = model(torch.tensor([corpus.word_to_ix[initial_text]]).view(1,-1), (state_h, state_c))
    for i in range(length):
        prev = np.argmax(pred.detach().view(1,-1))
        ind = prev.numpy().item()
        print(corpus.ix_to_word[ind],end=" ")
        pred, (state_h, state_c) = model(torch.tensor([prev]).view(1,-1), (state_h, state_c))

if __name__=="__main__":
    # model = TextGenerator(default.EMBEDDING_DIM,default.HIDDEN_LAYER_DIM,default.VOCAB_SIZE)
    # train(model,10)
    # save_model(model)
    model = load_model()
    generate_text(model,"the",30)