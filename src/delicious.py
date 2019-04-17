import json
import os
import re

import h5py
import numpy as np

from models.EncodeEstimator import EncodeEstimator
from utils import save_list, load, read_file, get_args


def read_vocabs(vocabs_path):
    vocabs = []
    with open(vocabs_path, encoding='utf-8') as f:
        for line in f:
            values = line.split(',')
            vocabs.append(values[0])
    return vocabs


def create_delicious_glove():
    embeddings_index = {}
    skipped_lines = []
    vocabs = read_vocabs('DeliciousMIL/vocabs.txt')
    with open('DeliciousMIL/glove.42B.300d.txt', encoding='utf-8') as f_42B:
        for i, line in enumerate(f_42B):
            values = line.split()
            word = values[0]
            if word not in vocabs:
                continue
            try:
                embedding = np.array([float(val) for val in values[1:]])
            except ValueError:
                print(i, word, values)
                skipped_lines.append(i)
                continue
            embeddings_index[word] = embedding

    with open('DeliciousMIL/glove.8520.300d.txt', mode='w', encoding='utf-8') as f_8520:
        for word in vocabs:
            if word not in embeddings_index:
                f_8520.write(word + '\n')  # todo
                continue
            f_8520.write(word + '\t')
            for x in embeddings_index[word]:
                f_8520.write(str(x) + ' ')
            f_8520.write('\n')


def reconstruct_txt_file_from_indices():
    vocabs = read_vocabs('DeliciousMIL/vocabs.txt')
    with open('DeliciousMIL/train-data.dat', encoding='utf-8') as f_dat:
        with open('DeliciousMIL/train-data.txt', mode='w', encoding='utf-8') as f_txt:
            for line in f_dat:
                indices = re.sub('[<>]', '', line).split()
                words = [vocabs[int(i)] for i in indices]
                f_txt.write(' '.join(words) + '\n')


def docs_gen(f):
    """

    :param f: embedded_sentences_file
    :return:
    """
    f_keys = sorted(list(f.keys()), key=int)

    for k in f_keys:
        if k == '100':
            break
        embedded_sentence = np.array(f.get(k))
        yield embedded_sentence, embedded_sentence.shape[0]


def main():
    # ---------
    # load data
    args = get_args()
    labels = load(args.labels_path)
    terms = load(args.terms_path)
    pub_med_ids, _ = read_file(args.documents_path)
    index2word = load(args.index2word_path)

    # ------
    # Encode
    params = vars(args)
    params['dropout'] = args.dropout
    params['data_size'] = len(labels)
    params['embedding_dim'] = args.embedding_dim
    params['num_epochs'] = args.num_epochs
    params['batch_size'] = args.batch_size
    params['term_size'] = args.mlp_layer_dims[-1]
    params['word_vecs_path'] = args.embedded_sentences.split('/')[1].split('.')[0]

    # get estimator
    estimator = EncodeEstimator(params)

    # todo
    out_dir = ""

    with h5py.File(args.embedded_sentences, 'r') as f:
        def sen_gen():
            for i in docs_gen(f):
                yield i[0]

        def len_gen():
            for i in docs_gen(f):
                yield i[1]

        if args.test_mode == 2:
            estimator.train(sen_gen, len_gen, labels, 1)
        else:
            estimator.train(sen_gen, len_gen, labels)

        doc_vecs, pred_labels = estimator.predict(sen_gen, len_gen)

    # ---------
    # save data
    # encoder data
    os.makedirs(out_dir)

    # write params to a txt file, except embeddings
    param_dir = out_dir + '/params.txt'
    with open(param_dir, 'w') as f:
        f.write(json.dumps(params))

    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]]
                              for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(out_dir + '/pred_labels.txt', pred_lab_words)


if __name__ == '__main__':
    # reconstruct_txt_file_from_indices()
    # exit()
    
    main()
