# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
import numpy as np
import pickle

from tagger_train import POSTagger


def read_test_data(data_path: str) -> list:
    testing_data = []
    with open(data_path, 'r') as fd:
        data = fd.readlines()
        for sentence in data:
            testing_data.append(sentence.split())

    return testing_data


def prepare_model_input(sent, word_to_ix, char_to_ix, pad='<P>'):
    words = [word_to_ix[w] if w in word_to_ix else word_to_ix['<UNK>'] for w in sent]
    chars = []
    max_word_len = max([len(w) for w in sent])

    for word in sent:
        word_chars = list(word)
        padding = [pad for _ in range(max_word_len - len(word_chars))]
        pad_word_chars = word_chars + padding
        chars.append([char_to_ix[c] for c in pad_word_chars])

    words_t = torch.tensor(words, dtype=torch.long).unsqueeze(0)
    chars_t = torch.tensor(chars, dtype=torch.long).unsqueeze(0)

    return words_t, chars_t


def tag_sentence(test_file, model_file, out_file):
    testing_data = read_test_data(test_file)
    word_to_ix, tag_to_ix, char_to_ix, params, model_state_dict = torch.load(model_file)
    ix_to_tag = dict((v, k) for (k, v) in tag_to_ix.items())
    pos_model = POSTagger(**params)  # 10, 46, 10, 38474, 10, 59, k=5, char_filters=10)
    pos_model.load_state_dict(model_state_dict)

    with open(out_file, 'w') as fd:
        result = []
        for i, sent in enumerate(testing_data):
            l_sent = [w.lower() for w in sent]
            words, chars = prepare_model_input(l_sent, word_to_ix, char_to_ix)
            res = torch.argmax(pos_model(words, chars), dim=2)
            out_sent = []
            for j, tag_id in enumerate(res.squeeze(0)):
                out_sent.append(testing_data[i][j] + "/" + ix_to_tag[tag_id.item()])

            result.append(" ".join(out_sent))

        fd.write("\n".join(result))

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
