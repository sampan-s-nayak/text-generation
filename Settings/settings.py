""" 
    this file contains the configuration settings for the neural network
"""
import Data_loader.dataloader as dl

# PATH = 'Corpus/bootstrap_code.txt'
PATH = 'Corpus/shakespeare.txt'

EMBEDDING_DIM = 50

HIDDEN_LAYER_DIM = 120

# gradient clipping parameter
GRADIENTS_NORM = 2

VOCAB_SIZE = dl.TextCorpus().num_vocab

# the sequence will not be of fixed size every time hence if we want to change batch size to a value other than 1 then padding is required (refer: https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068)
BATCH_SIZE = 120

DEPLOY_BATCH_SIZE = 1

LEARNING_RATE = 0.0001

WEIGHTS_PATH = "Weights/"