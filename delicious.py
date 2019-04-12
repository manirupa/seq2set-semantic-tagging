import numpy as np
import re
from utils import save, save_list, load
import json
import h5py
import os


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


def encode_nn(
        docs_path,
        labels,
        model,
        test_mode,
        loss_fn,
        term_size,
        kln):

    model_params, estimator_params = get_params(model)
    model_params['mlp_layer_dims'] += [term_size]
    model_params['dropout'] = FLAGS.dropout
    model_params['kln'] = kln  # kln: k-label-neighbor

    if 'num_layers' not in model_params:
        model_params['num_layers'] = 'none'

    estimator_params['data_size'] = len(labels)
    estimator_params['embedding_dim'] = FLAGS.embedding_dim
    estimator_params['num_epochs'] = FLAGS.num_epochs
    estimator_params['batch_size'] = FLAGS.batch_size
    estimator_params['term_size'] = term_size
    model_params['word_vecs_path'] = FLAGS.embedded_sentences.split('/')[1].split('.')[0]
    estimator_params['folder'] = None

    # get estimator
    estimator = EncodeEstimatorFactory.get_estimator(
        loss_fn, model, estimator_params, model_params)

    with h5py.File(docs_path, 'r') as f:
        def sen_gen():
            for i in docs_gen(f):
                yield i[0]

        def len_gen():
            for i in docs_gen(f):
                yield i[1]

        if test_mode == 2:
            estimator.train(sen_gen, len_gen, labels, 1)
        else:
            estimator.train(sen_gen, len_gen, labels)

        doc_vecs, pred_labels = estimator.predict(sen_gen, len_gen)

    all_params = {**model_params, **estimator_params}
    return doc_vecs, pred_labels, estimator.out_dir, all_params


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


def main(_):
    # ---------
    # load data
    labels = load(FLAGS.labels_path)
    terms = load(FLAGS.terms_path)
    pub_med_ids, _ = read_file(FLAGS.documents_path)
    index2word = load(FLAGS.index2word_path)

    # --------------------
    # Encode
    doc_vecs, pred_labels, out_dir, hparams = encode_nn(
        FLAGS.embedded_sentences,
        labels,
        FLAGS.model,
        FLAGS.test_mode,
        FLAGS.loss_fn,
        len(terms),
        FLAGS.labels_path.split('/')[1])

    # ---------
    # save data
    # encoder data
    os.makedirs(out_dir)

    # write params to a txt file, except embeddings
    param_dir = out_dir + '/params.txt'
    with open(param_dir, 'w') as f:
        f.write(json.dumps(hparams))

    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]]
                              for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(out_dir + '/pred_labels.txt', pred_lab_words)


if __name__ == '__main__':
    reconstruct_txt_file_from_indices()
    print(0)
