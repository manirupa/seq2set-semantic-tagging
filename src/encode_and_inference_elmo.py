import json
import os
import time

import h5py
import numpy as np
import tensorflow as tf

from models import inference
from models.EncodeEstimatorElmo import EncodeEstimator
from utils import load, save, save_list, read_file, get_args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_float("dropout", 0.2, "dropout rate")
flags.DEFINE_integer(
    "k", 5, "top k closest docs of a query doc")
flags.DEFINE_integer(
    "num_epochs", 60, " ")
flags.DEFINE_integer(
    "batch_size", 64, " ")
flags.DEFINE_integer(
    "embedding_dim", 50, " ")
flags.DEFINE_integer(
    "test_mode", 0,
    "if 0, normal."
    "if 1, inference will only calculate the cosine"
    "similarities of the first 100 docs."
    "if 2, inference will only calculate the cosine"
    "similarities of the first 100 docs and Encoder"
    "will only train for 1 step.")
flags.DEFINE_string(
    "model", "BiLSTM",
    "options: DAN, LSTM, BiLSTM, BiLSTMATT,"
    "Transformer, doc2vec")
flags.DEFINE_string(
    "loss_fn", "sigmoid",
    "options: softmax_uniform, softmax_skewed_labels, lm,"
    "sigmoid, sigmoid_with_constraint")
flags.DEFINE_string(
    "embedded_sentences", "data/elmo_50_sentences.hdf5", " ")
flags.DEFINE_string(
    "labels_path", "data/labels.pickle", " ")
flags.DEFINE_string(
    "doc_tfidf_reps_path", "data/doc_tfidf_reps_mc1.pickle", " ")
flags.DEFINE_string(
    "index2word_path", "data/index2word_mc1.pickle", " ")
flags.DEFINE_string(
    "terms_path", "data/terms.pickle", " ")
flags.DEFINE_string(
    "documents_path", "data/cleaned.txt", " ")  # data/cleaned_phrase_embedded.txt
flags.DEFINE_string(
    "fuse_doc_type", "arithmetic_mean",
    "options: arithmetic_mean, geometric_mean")
FLAGS = flags.FLAGS


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
    # Load data
    params = vars(get_args())
    # todo remove samples with empty labels for softmax
    labels = load(FLAGS.labels_path)
    terms = load(FLAGS.terms_path)

    # get params
    params['pred_data_size'] = len(labels)
    params['embeddings_dim'] = FLAGS.embedding_dim
    params['dropout'] = FLAGS.dropout
    params['num_epochs'] = FLAGS.num_epochs
    params['batch_size'] = FLAGS.batch_size
    params['term_size'] = len(terms)
    params['loss_fn'] = FLAGS.loss_fn
    params['model'] = FLAGS.model

    folder = '%d_%s_%s_%s_nl%s_kln%s_dp%s_ep%d_bs%d' % (
        int(time.time()), params['model'], params['loss_fn'],
        os.path.split(FLAGS.embedded_sentences)[1].split('.')[0],
        params['num_layers'], os.path.split(FLAGS.labels_path)[1],
        params['dropout'], params['num_epochs'], params['batch_size'])

    params['model_dir'] = os.path.join("results", "models", folder)
    out_dir = os.path.join("results", "outputs", folder)
    os.makedirs(out_dir, exist_ok=True)

    # ------
    # Encode
    estimator = EncodeEstimator(params)

    with h5py.File(FLAGS.embedded_sentences, 'r') as f:
        def sen_gen():
            for i in docs_gen(f):
                yield i[0]

        def len_gen():
            for i in docs_gen(f):
                yield i[1]

        if FLAGS.test_mode == 2:
            estimator.train(sen_gen, len_gen, labels, 1)
        else:
            estimator.train(sen_gen, len_gen, labels)

        doc_vecs, pred_labels = estimator.predict(sen_gen, len_gen)

    # write params to a txt file, except embeddings
    param_dir = os.path.join(out_dir, 'params.txt')
    if 'embeddings' in params:
        del params['embeddings']
    with open(param_dir, 'w') as f:
        f.write(json.dumps(params))
    del params

    # ---------
    # Inference
    doc_tfidf_reps = labels
    if len(FLAGS.doc_tfidf_reps_path) > 0:
        doc_tfidf_reps = load(FLAGS.doc_tfidf_reps_path)

    fused_docs, expanded, top_k_indices = inference.main(
        doc_vecs, doc_tfidf_reps, FLAGS.k, FLAGS.fuse_doc_type)

    save(os.path.join(out_dir, 'top_k_indices'), top_k_indices)
    np.save(os.path.join(out_dir, 'fused_docs'), fused_docs)
    np.save(os.path.join(out_dir, 'doc_vecs'), doc_vecs)
    del doc_vecs, top_k_indices, fused_docs

    # ---------
    # Save data
    pub_med_ids, _ = read_file(FLAGS.documents_path)
    index2word = load(FLAGS.index2word_path)

    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]] for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(os.path.join(out_dir, 'pred_labels.txt'), pred_lab_words)

    # convert to word ids
    labels = [[terms[l] for l in lab] for lab in labels]

    if len(FLAGS.doc_tfidf_reps_path) == 0:
        expanded = [[terms[l] for l in lab] for lab in expanded]

    expanded_labels = []
    for p_id, l, ex in zip(pub_med_ids, labels, expanded):
        e_words = ', '.join([index2word[e] for e in ex])
        original = ', '.join([index2word[i] for i in l])
        line = str(p_id) + '\tORIGINAL: ' + original + '\tEXPANDED: ' + e_words
        expanded_labels.append(line)

    fname = os.path.split(out_dir)[-1] + '_expanded_labels.txt'
    expanded_labels_dir = os.path.join(out_dir, fname)
    save_list(expanded_labels_dir, expanded_labels)


def convert_lists(labels, pred_labels, expanded, index2word, terms, is_doc_tfidf_reps):
    # convert to word ids
    labels = [[terms[l] for l in lab] for lab in labels]
    pred_labels = [[terms[l] for l in lab] for lab in pred_labels]

    if not is_doc_tfidf_reps:
        expanded = [[terms[l] for l in lab] for lab in expanded]

    # convert to str
    labels = [[index2word[l] for l in lab] for lab in labels]
    pred_labels = [[index2word[l] for l in lab] for lab in pred_labels]
    expanded = [[index2word[l] for l in lab] for lab in expanded]

    return labels, pred_labels, expanded


def write_results_to_file(out_dir, pub_med_ids, labels, pred_labels, expanded):
    expanded_labels = []
    for p_id, lab, pl, ex in zip(pub_med_ids, labels, pred_labels, expanded):
        orig = ', '.join(lab)
        pred_lab = ', '.join(pl)
        e_lab = ', '.join(ex)
        line = str(p_id) + '\tORIGINAL: ' + orig + '\tPREDICTED' + pred_lab + '\tEXPANDED: ' + e_lab
        expanded_labels.append(line)

    fname = os.path.split(out_dir)[-1] + '_expanded_labels.txt'
    expanded_labels_dir = os.path.join(out_dir, fname)
    save_list(expanded_labels_dir, expanded_labels)


if __name__ == '__main__':
    tf.app.run()
