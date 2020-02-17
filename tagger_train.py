# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class POSTagger(nn.Module):

    def __init__(self, hidden_dim, target_size, w_embedding_dim,
                 w_vocab_size, c_embedding_dim, c_vocab_size):
        super(POSTagger, self).__init__()
        # TODO: Add more params
        self.lstm = nn.LSTM(hidden_size=hidden_dim, bidirectional=True)
        self.cnn = nn.Sequential(
            # TODO: define structure
        )
        self.word_embeddings = nn.Embedding(w_vocab_size, w_embedding_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        w_embeddings = self.word_embeddings(sentence)
        c_embeddings = self.cnn(sentence)

        full_embedding = torch.Tensor()  # TODO: concat word and character embeddings

        lstm_out, _ = self.lstm(full_embedding)
        tag_space = self.hidden2tag(lstm_out)
        # TODO: Do we need this? Cross-entropy uses softmax under the hood (IINM)
        # tag_scores = F.log_softmax(tag space)
        return tag_space


def read_train_data(data_path: str) -> list:
    training_data = []
    with open(data_path, 'r') as fd:
        data = fd.readlines()
        for sentence in data:
            words2tokens = []
            for tagged_word in sentence.split():
                # TODO: Create tuples for each element in sequence of / ???
                word, tag = tuple(tagged_word.rsplit('/', 1))
                words2tokens.append((word.lower(), tag))
            
            training_data.append(tuple(map(list, zip(*words2tokens))))
    
    return training_data


def get_word_tag_mappings(training_data: list) -> (dict, dict):
    word_to_ix = {}
    tag_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


def train_model(train_file, model_file):
    # Write your code here. You can add functions as well. Use torch 
    # library to save model parameters, hyperparameters, etc. to model_file
    EMBEDDING_DIM, HIDDEN_DIM = 7, 7
    training_data = read_train_data(train_file)
    word_to_ix, tag_to_ix = get_word_tag_mappings(training_data)

    pos_model = POSTagger()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pos_model.parameters(), lr=0.001)


    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
