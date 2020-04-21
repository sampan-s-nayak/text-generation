""" 
    this file contains the configuration settings for the neural network
"""
import Data_loader.dataloader as dl
import dataset

EMBEDDING_DIM = 50

HIDDEN_LAYER_DIM = 120

# gradient clipping parameter
GRADIENTS_NORM = 2

CORPUS = dl.TextCorpus(path=dataset.PATH)

VOCAB_SIZE = CORPUS.num_vocab

# the sequence will not be of fixed size every time hence if we want to change batch size to a value other than 1 then padding is required (I havent used padding) (refer: https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068)
BATCH_SIZE = 120

DEPLOY_BATCH_SIZE = 1

LEARNING_RATE = 0.0001

WEIGHTS_PATH = "Weights/"

DEVICE = CORPUS.device