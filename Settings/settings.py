""" 
    this file contains the configuration settings for the neural network
"""
import Data_loader.dataloader as dl

PATH = 'Corpus/The_Tempest.txt'

EMBEDDING_DIM = 30

HIDDEN_LAYER_DIM = 20

# gradient clipping parameter
GRADIENTS_NORM = 5

VOCAB_SIZE = dl.TextCorpus().num_vocab

# the sequence will not be of fixed size every time hence if we want to change batch size to a value other than 1 then padding is required (refer: https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068)
BATCH_SIZE = 1

LEARNING_RATE = 0.001

WEIGHTS_PATH = "Weights/checkpoint1.pth"