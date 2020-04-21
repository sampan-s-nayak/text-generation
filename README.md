# text-generation
A **word** level LSTM network trained to generate text.The neural network is written in pytorch and is deployed using a flask REST api.

## Execution
1. clone this repo
2. in the terminal,execute
```
cd text-generation
FLASK_ENV=development FLASK_APP=api.py flask run
```
3. open index.html in your browser

## Dataset
pretrained weights for two datasets have been provided
1. Shakespeare text corpus [link](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)
2. Bootstrap code [link](https://getbootstrap.com/)
<br />
note: both the networks were trained for 25 epochs,the inference weights being used can be changed in 
Settings/dataset.py

## Front End
![front end screen shot](https://github.com/saoalo/text-generation/blob/master/screen%20shots/frontend.png)
front end for the flask api designed using bootstrap,html and javascript
![sample output](https://github.com/saoalo/text-generation/blob/master/screen%20shots/example.png)
output of the network trained on shakespeare dataset

### Sample output for bootstrap dataset 
```
input:
.display-1 { 

output:
.display-1 { 
  display : # fff ; ; 
  flex : ; 
  background-color ! important ! ,
  { 
    display { 
      color ;
      : rgba 0 , ;
    } 
    { 
      margin-left 
      { color ; 
        display { 
        display : 0 , ! ! ! important
```
we can see that the network has learnt that an attribute follows after a '{' and that a ':' follows after the attribute.The network
also learns the semantics of using ';' 

## Requirements
```
Pytorch
Bootstrap(uses cdn)
Flask
NLTK
```
## Dataloader
A TextLoader class which inherits from Pytoch dataset class has been created to handle all of the text processing required,
```python
class TextCorpus(Dataset):
    def __init__(self, path=ds.PATH):
        # choose device where the data will reside
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # read the corpus
        with open(path,'r') as corpus:
            self.data = corpus.read()

        # tokenize the text
        self.word_token = nltk.word_tokenize(self.data.lower())

        # generate the vocabulary
        self.vocab = sorted(set(self.word_token))
        self.num_vocab = len(self.vocab)

        # generate a mapping from words to indices in a dictionary (and vice-versa)
        self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.ix_to_word = {i:word for i,word in enumerate(self.vocab)}


    def __getitem__(self, idx):
        x = torch.tensor([self.word_to_ix[self.word_token[idx]]],dtype=torch.long,device=self.device)
        y = torch.tensor([self.word_to_ix[self.word_token[idx+1]]],dtype=torch.long,device=self.device)
        data = {
            "curr_word":x,
            "next_word":y
        }
        return data

    def __len__(self):
        return len(self.word_token)-1
```

## Network Parameters
The network parameters used can be changed in `Settings/settings.py` file
snap shot of settings used: 
```
EMBEDDING_DIM = 50
HIDDEN_LAYER_DIM = 120
GRADIENTS_NORM = 2
CORPUS = dl.TextCorpus(path=dataset.PATH)
VOCAB_SIZE = CORPUS.num_vocab
BATCH_SIZE = 120
DEPLOY_BATCH_SIZE = 1
LEARNING_RATE = 0.0001
WEIGHTS_PATH = "Weights/"
DEVICE = CORPUS.device
```

## Network Snapshot
```python
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
        return (torch.zeros(1, batch_size, self.hidden_dim).to(default.DEVICE),
                torch.zeros(1, batch_size, self.hidden_dim).to(default.DEVICE))
```

<br />

## Training Loop

```python
def train(model,epoch,init_trained=0):
    loss_fn,optimizer = get_loss_and_optimizer(model)
    dataloader = DataLoader(default.CORPUS, batch_size=default.BATCH_SIZE,shuffle=False)
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
```

<br />

## Prediction

```python
def generate_text(model,initial_text,length):
    output = ""
    
    # setting the model to eval mode
    model.eval()
    
    # feeding the initial text to the model after processing
    state_h, state_c = model.zero_state(default.DEPLOY_BATCH_SIZE)
    tokens = nltk.word_tokenize(initial_text)
    for token in tokens:
        output = output + str(token) + " "
        pred, (state_h, state_c) = model(torch.tensor([default.CORPUS.word_to_ix[token.lower()]],device=default.DEVICE).view(1,-1), (state_h, state_c))
    
    # generating text
    for i in range(length):
        _, top_ix = torch.topk(pred[0], k=4)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        output = output + str(default.CORPUS.ix_to_word[choice]) + " "
        pred, (state_h, state_c) = model(torch.tensor([choice],device=default.DEVICE).view(1,-1), (state_h, state_c))
    return output
```

## References
1. course on sequence models [link](https://www.coursera.org/learn/nlp-sequence-models)
2. A tutorial on training an lstm to generate text using pytorch [link](https://machinetalk.org/2019/02/08/text-generation-with-pytorch/)
3. tutorial in tensorflow (character level LSTM) [link](https://www.tensorflow.org/tutorials/text/text_generation)
4. blog on RNN's [link](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
5. tutorial on deploying pytorch models as a REST api in flask [link](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
6. tutorial on using word embeddings in pytorch [link](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
