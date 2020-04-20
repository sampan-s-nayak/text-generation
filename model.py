import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from Data_loader import dataloader as dl
from Settings import settings as default
import nltk

# seed = 0
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

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
        return (torch.zeros(1, batch_size, self.hidden_dim).to(corpus.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(corpus.device))

def get_loss_and_optimizer(model,learning_rate=default.LEARNING_RATE):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def train(model,epoch,init_trained=0):
    loss_fn,optimizer = get_loss_and_optimizer(model)
    dataloader = DataLoader(corpus, batch_size=default.BATCH_SIZE,shuffle=False)
    for i in range(epoch - init_trained):
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

            loss_value = loss.detach().item()

            # Perform back-propagation
            loss.backward()

            # perform gradient clipping to prevent gradients from exploding
            _ = torch.nn.utils.clip_grad_norm_(
                model.parameters(), default.GRADIENTS_NORM)

            # Update the network's parameters
            optimizer.step()

            epoch_loss += loss_value
        print(f"epoch: {i+1+init_trained} loss: {loss}")
        save_model(model,path=os.path.join(default.WEIGHTS_PATH,f"bootstrap_epoch({i+1+init_trained})_loss({loss}).pth"))
    return model

def save_model(model,path=os.path.join(default.WEIGHTS_PATH,"checkpoint1.pth")):
    torch.save(model.state_dict(), path)

def load_model(path=os.path.join(default.WEIGHTS_PATH,"checkpoint1.pth")):
    model = TextGenerator(default.EMBEDDING_DIM,default.HIDDEN_LAYER_DIM,default.VOCAB_SIZE)
    model.load_state_dict(torch.load(path,map_location=torch.device(corpus.device)))
    model.eval()
    return model

def generate_text(model,initial_text,length):
    # initial LSTM stage
    model.eval()
    state_h, state_c = model.zero_state(default.DEPLOY_BATCH_SIZE)
    tokens = nltk.word_tokenize(initial_text)
    for token in tokens:
        print(token,end=" ")
        pred, (state_h, state_c) = model(torch.tensor([corpus.word_to_ix[token.lower()]],device=corpus.device).view(1,-1), (state_h, state_c))
    for i in range(length):
        _, top_ix = torch.topk(pred[0], k=10)
        choices = top_ix.tolist()
        ind = np.random.choice(choices[0])
        print(corpus.ix_to_word[ind],end=" ")
        pred, (state_h, state_c) = model(torch.tensor([ind],device=corpus.device).view(1,-1), (state_h, state_c))

if __name__=="__main__":
    # model = TextGenerator(default.EMBEDDING_DIM,default.HIDDEN_LAYER_DIM,default.VOCAB_SIZE)
    # model.to(corpus.device)
    # model = train(model,25)

    model = load_model(path=os.path.join(default.WEIGHTS_PATH,'shakespeare_epoch(25)_loss(4.941622734069824).pth'))
    model.to(corpus.device)
    generate_text(model,initial_text="in sooth",length=400)