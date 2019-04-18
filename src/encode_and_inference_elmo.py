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
    args = get_args()
    params = vars(args)
    # todo remove samples with empty labels for softmax
    labels = load(args.labels_path)
    terms = load(args.terms_path)

    # get params
    params['pred_data_size'] = len(labels)
    params['embeddings_dim'] = args.embedding_dim

    folder = '%d_%s_%s_%s_nl%s_kln%s_dp%s_ep%d_bs%d' % (
        int(time.time()), params['model'], params['loss_fn'],
        os.path.split(args.embedded_sentences)[1].split('.')[0],
        params['num_layers'], os.path.split(args.labels_path)[1],
        params['dropout'], params['num_epochs'], params['batch_size'])

    params['model_dir'] = os.path.join("results", "models", folder)
    out_dir = os.path.join("results", "outputs", folder)
    os.makedirs(out_dir, exist_ok=True)

    # ------
    # Encode
    estimator = EncodeEstimator(params)

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
    if len(args.doc_tfidf_reps_path) > 0:
        doc_tfidf_reps = load(args.doc_tfidf_reps_path)

    fused_docs, expanded, top_k_indices = inference.main(
        doc_vecs, doc_tfidf_reps, args.k, args.fuse_doc_type)

    save(os.path.join(out_dir, 'top_k_indices'), top_k_indices)
    np.save(os.path.join(out_dir, 'fused_docs'), fused_docs)
    np.save(os.path.join(out_dir, 'doc_vecs'), doc_vecs)
    del doc_vecs, top_k_indices, fused_docs

    # ---------
    # Save data
    pub_med_ids, _ = read_file(args.documents_path)
    index2word = load(args.index2word_path)

    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]] for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(os.path.join(out_dir, 'pred_labels.txt'), pred_lab_words)

    # convert to word ids
    labels = [[terms[l] for l in lab] for lab in labels]

    if len(args.doc_tfidf_reps_path) == 0:
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
