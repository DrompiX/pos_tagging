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
        self.char_conv = nn.Conv1d(in_channels=c_embedding_dim, out_channels=char_filters,
                                   kernel_size=k, padding=1) # maybe use (k - 1) // 2 ?

        # LSTM
        self.lstm = nn.LSTM(w_embedding_dim + char_filters, hidden_dim, bidirectional=True)

        # Linear mapping
        self.hidden2tag = nn.Linear(2 * hidden_dim, target_size)

    def forward(self, word_batch, char_batch):
        w_embeddings = self.word_embeddings(word_batch)
        c_embeddings = self.char_embeddings(char_batch)

        cB, cW, cC, cE = c_embeddings.shape
        c_embeddings = c_embeddings.transpose(3, 2).reshape((cB * cW, cE, cC))
        cv = self.char_conv(self.dropout(c_embeddings))
        pool = F.max_pool1d(cv, cv.shape[2]).squeeze(2)
        char_repr = pool.reshape(cB, cW, -1)

        full_embedding = torch.cat((w_embeddings, char_repr), dim=2)

        lstm_out, _ = self.lstm(full_embedding)

        tag_space = self.hidden2tag(lstm_out)
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
    word_to_ix = {"<P>": 0}  # padding word
    tag_to_ix = {"<P>": 0}  # padding tag
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


def get_char_vocabulary(training_data: list) -> dict:
    vocab = set()
    vocab.add("<P>")  # ~ - padding character
    for words, _ in training_data:
        vocab |= set("".join(words))
    return dict((k, v) for k, v in zip(sorted(list(vocab)), range(len(vocab))))


def prepare_sequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def batch_generator(data, word_to_ix, char_to_ix, tag_to_ix, batch_size, pad="<P>"):
    ord_data = sorted(data, key=lambda x: len(x[0]))
    for i in range(0, int(np.ceil(len(ord_data) / batch_size))):
        start_id = i * batch_size
        end_id = start_id + batch_size
        batch_slice = ord_data[start_id: end_id]
        max_len = len(batch_slice[-1][0])

        word_batch = []
        char_batch = []
        tags_batch = []
        max_word_len = max([len(w) for sent, _ in batch_slice for w in sent])
        for sent, tags in batch_slice:
            padding = [pad for _ in range(max_len - len(sent))]
            pad_sent = sent + padding
            pad_tags = tags + padding
            word_batch.append([word_to_ix[w] for w in pad_sent])
            tags_batch.append([tag_to_ix[w] for w in pad_tags])

            sent_char_batch = []
            for word in pad_sent:
                word_chars = [pad] if word == pad else list(word)
                padding = [pad for _ in range(max_word_len - len(word_chars))]
                pad_word_chars = word_chars + padding
                sent_char_batch.append([char_to_ix[c] for c in pad_word_chars])

            char_batch.append(sent_char_batch)

        word_batch_t = torch.tensor(word_batch, dtype=torch.long)
        char_batch_t = torch.tensor(char_batch, dtype=torch.long)
        tags_batch_t = torch.tensor(tags_batch, dtype=torch.long)

        yield word_batch_t, char_batch_t, tags_batch_t


def train_epoch(pos_model, batches, criterion, optimizer):
    for i, (word_batch, char_batch, tags_batch) in enumerate(batches):
        print(f'\r-> Batch [{i + 1}]', end='')
        tag_space = pos_model(word_batch, char_batch)


def train_model(train_file, model_file):
    # Write your code here. You can add functions as well. Use torch 
    # library to save model parameters, hyperparameters, etc. to model_file
    EMBEDDING_DIM, HIDDEN_DIM = 7, 7
    training_data = read_train_data(train_file)
    word_to_ix, tag_to_ix = get_word_tag_mappings(training_data)
    ix_to_word = dict((v, k) for (k, v) in word_to_ix.items())
    ix_to_tag = dict((v, k) for (k, v) in tag_to_ix.items())
    char_to_ix = get_char_vocabulary(training_data)

    pos_model = POSTagger(HIDDEN_DIM, len(tag_to_ix.keys()), EMBEDDING_DIM, len(word_to_ix.keys()),
                          7, len(char_to_ix))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pos_model.parameters(), lr=0.001)

    batches = list(batch_generator(training_data, word_to_ix, char_to_ix, tag_to_ix, 5))

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        train_epoch(pos_model, batches, criterion, optimizer)
        print()

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
