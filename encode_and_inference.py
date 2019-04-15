import json
import os
import random
import shutil
import time

import numpy as np
import tensorflow as tf

from model_params import get_params
from models import inference
from models.EncodeEstimator import EncodeEstimator
from utils import load, save, save_list, read_file, remove_samples_with_empty_labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_float("dropout", 0.2, "dropout rate")
flags.DEFINE_float("alpha", 1, "weight of LM loss")
flags.DEFINE_integer("k", 5, "top k closest docs of a query doc")
flags.DEFINE_integer("num_epochs", 60, " ")
flags.DEFINE_integer("batch_size", 64, " ")
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
    "options: DAN, LSTM, BiLSTM, BiLSTMATT, Transformer, doc2vec")
flags.DEFINE_string(
    "loss_fn", "sigmoid",
    "options: softmax_uniform, softmax_skewed_labels, lm,"
    "sigmoid, sigmoid_with_constraint")
flags.DEFINE_string("word_vecs_path", "data/word2vec_sg1_s50_w4_m1_n5_i15.npy", " ")
flags.DEFINE_string("docs_path", "data/docs_word_indices_mc1.pickle", " ")
flags.DEFINE_string("labels_path", "data/labels.pickle", " ")
flags.DEFINE_string("doc_tfidf_reps_path", "data/doc_tfidf_reps_mc1.pickle", " ")
flags.DEFINE_string("index2word_path", "data/index2word_mc1.pickle", " ")
flags.DEFINE_string("terms_path", "data/terms.pickle", " ")
flags.DEFINE_string("documents_path", "data/cleaned.txt", " ")  # data/cleaned_phrase_embedded.txt
flags.DEFINE_string("fuse_doc_type", "arithmetic_mean", "options: arithmetic_mean, geometric_mean")
flags.DEFINE_string(
    'folder', '1551239956_BiLSTM__SigmoidEncode_word2vec_sg1_s50_w4_m1_n5_i15.npy_nl4_klnlabels.pickle_dp0.2_ep1_bs32',
    'model folder')
tf.flags.DEFINE_boolean('keep_model_files', True, " ")
tf.flags.DEFINE_boolean('direct', False, " ")
FLAGS = flags.FLAGS


def main(_):
    # ---------
    # Load data
    docs = load(FLAGS.docs_path)
    labels = load(FLAGS.labels_path)
    terms = load(FLAGS.terms_path)

    # todo: remove this?
    if FLAGS.loss_fn == 'softmax_uniform' or FLAGS.loss_fn == 'softmax_skewed_labels':
        docs, labels = remove_samples_with_empty_labels(docs, labels)

    if FLAGS.test_mode != 0:
        docs = docs[:100]
        labels = labels[:100]

    zipped = list(zip(docs, labels))
    random.seed(42)
    random.shuffle(zipped)
    training_set_size = int(len(zipped)*0.9)
    training_set = zipped[:training_set_size]
    docs_train, labels_train = [[*x] for x in zip(*training_set)]
    eval_set = zipped[training_set_size:]
    docs_eval, labels_eval = [[*x] for x in zip(*eval_set)]
    labels_lm_train = [[terms[l] for l in lab] for lab in labels_train]
    labels_lm_eval = [[terms[l] for l in lab] for lab in labels_eval]

    # get params
    params = get_params(FLAGS.model)
    params['embeddings'] = np.load(FLAGS.word_vecs_path)
    params['embeddings_dim'] = params['embeddings'].shape[1]
    params['dropout'] = FLAGS.dropout
    params['num_epochs'] = FLAGS.num_epochs
    params['batch_size'] = FLAGS.batch_size
    params['term_size'] = len(terms)
    params['loss_fn'] = FLAGS.loss_fn
    params['model'] = FLAGS.model
    params['alpha'] = FLAGS.alpha

    if FLAGS.direct:
        folder = FLAGS.folder
        params['model_dir'] = os.path.join("results", "models", folder)
        out_dir = os.path.join("results", "outputs", folder, "direct")
    else:
        folder = '%d_%s_%s_%s_nl%s_kln%s_dp%s_ep%d_bs%d' % (
            int(time.time()), params['model'], params['loss_fn'],
            os.path.split(FLAGS.word_vecs_path)[-1],
            params['num_layers'], os.path.split(FLAGS.labels_path)[1],
            params['dropout'], params['num_epochs'], params['batch_size'])
        params['model_dir'] = os.path.join("results", "models", folder)
        out_dir = os.path.join("results", "outputs", folder)
    os.makedirs(out_dir, exist_ok=True)

    # ------
    # Encode
    estimator = EncodeEstimator(params)

    if not FLAGS.direct:
        if FLAGS.test_mode == 2:
            estimator.train_and_eval(
                docs_train, labels_lm_train, labels_train,
                docs_eval, labels_lm_eval, labels_eval, max_step=1)
        else:
            estimator.train_and_eval(
                docs_train, labels_lm_train, labels_train,
                docs_eval, labels_lm_eval, labels_eval)

    estimator.batch_size = 128  # takes less time with large batch size
    doc_vecs, pred_labels = estimator.predict(docs)

    pub_med_ids, _ = read_file(FLAGS.documents_path)
    index2word = load(FLAGS.index2word_path)

    # save predicted labels to disk
    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]] for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(os.path.join(out_dir, 'pred_labels.txt'), pred_lab_words)

    if not FLAGS.keep_model_files:
        shutil.rmtree(params['model_dir'], ignore_errors=True)

    # write params to a txt file, except embeddings
    param_dir = os.path.join(out_dir, 'params.txt')
    if 'embeddings' in params:
        del params['embeddings']
    with open(param_dir, 'w') as f:
        f.write(json.dumps(params))
    del params

    print("Finished predicting.")
    exit(1)

    # ---------
    # Inference
    doc_tfidf_reps = labels
    if len(FLAGS.doc_tfidf_reps_path) > 0:
        doc_tfidf_reps = load(FLAGS.doc_tfidf_reps_path)

    fused_docs, expanded, top_k_indices = inference.main(
        doc_vecs, doc_tfidf_reps, FLAGS.k, FLAGS.fuse_doc_type)

    save(os.path.join(out_dir, 'top_k_indices'), top_k_indices)
    # np.save(os.path.join(out_dir, 'fused_docs'), fused_docs)
    # np.save(os.path.join(out_dir, 'doc_vecs'), doc_vecs)
    del doc_vecs, top_k_indices, fused_docs

    # ----------------------------
    # Save expanded labels to disk
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
