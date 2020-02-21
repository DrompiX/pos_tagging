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
                 w_vocab_size, c_embedding_dim, c_vocab_size, k=3, char_filters=10):
        super(POSTagger, self).__init__()
        # Embeddings
        self.word_embeddings = nn.Embedding(w_vocab_size, w_embedding_dim)
        self.char_embeddings = nn.Embedding(c_vocab_size, c_embedding_dim)

        # CNN
        self.dropout = nn.Dropout(0.2)
        self.char_conv = nn.Conv1d(in_channels=c_embedding_dim, out_channels=char_filters, kernel_size=k)
        # self.pool = F.max_pool1d()

        # LSTM
        # self.lstm = nn.LSTM(hidden_size=hidden_dim, bidirectional=True)

        # Linear mapping
        # self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        w_embeddings = self.word_embeddings(sentence)
        # c_embeddings = self.char_embeddings(sentence)
        print(w_embeddings.shape)
        # print(c_embeddings.shape)

        # TODO: make some call to cnn

        full_embedding = torch.Tensor()  # TODO: concat word and character embeddings

        # lstm_out, _ = self.lstm(full_embedding)
        # tag_space = self.hidden2tag(lstm_out)
        # TODO: Do we need this? Cross-entropy uses softmax under the hood (IINM)
        # tag_scores = F.log_softmax(tag space)
        # return tag_space


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
    word_to_ix = {"<e>": 0}  # padding word
    tag_to_ix = {"<E>": 0}  # padding tag
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


def get_char_vocabulary(training_data: list) -> dict:
    vocab = set("~")  # ~ - padding character
    for words, _ in training_data:
        vocab |= set("".join(words))
    return dict((k, v) for k, v in zip(sorted(list(vocab)), range(len(vocab))))


def prepare_sequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def batch_generator(data, word_to_ix, batch_size):
    ord_data = sorted(data, key=lambda x: len(x[0]))
    for i in range(0, int(np.ceil(len(ord_data) / batch_size))):
        start_id = i * batch_size
        end_id = start_id + batch_size
        batch_slice = ord_data[start_id: end_id]
        max_len = len(batch_slice[-1][0])

        batch = []
        for sent, _ in batch_slice:
            if len(sent) < max_len:
                ext_sent = sent + ["<e>" for _ in range(max_len - len(sent))]
            else:
                ext_sent = sent
            batch.append([word_to_ix[w] for w in ext_sent])

        yield torch.tensor(batch, dtype=torch.long)


def train_model(train_file, model_file):
    # Write your code here. You can add functions as well. Use torch 
    # library to save model parameters, hyperparameters, etc. to model_file
    EMBEDDING_DIM, HIDDEN_DIM = 7, 7
    training_data = read_train_data(train_file)
    word_to_ix, tag_to_ix = get_word_tag_mappings(training_data)
    char_vocab = get_char_vocabulary(training_data)

    pos_model = POSTagger(HIDDEN_DIM, len(tag_to_ix.keys()), EMBEDDING_DIM, len(word_to_ix.keys()),
                          7, len(char_vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pos_model.parameters(), lr=0.001)

    for batch in batch_generator(training_data, word_to_ix, 5):
        print('Shape:', batch.shape)
        pos_model(batch)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
