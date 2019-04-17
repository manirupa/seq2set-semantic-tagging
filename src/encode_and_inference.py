import json
import os
import random
import shutil
import time

import numpy as np
import tensorflow as tf

from models import inference
from models.EncodeEstimator import EncodeEstimator
from utils import load, save, save_list, read_file, remove_samples_with_empty_labels, get_args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)


def main():
    # ---------
    # Load data
    args = get_args()
    docs = load(args.docs_path)
    labels = load(args.labels_path)
    terms = load(args.terms_path)

    if args.loss_fn == 'softmax_uniform' or args.loss_fn == 'softmax_skewed_labels':
        docs, labels = remove_samples_with_empty_labels(docs, labels)

    if args.test_mode != 0:
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
    params = vars(args)
    params['embeddings'] = np.load(args.word_vecs_path)
    params['embeddings_dim'] = params['embeddings'].shape[1]

    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if args.folder is not None:
        folder = args.folder
        params['model_dir'] = os.path.join(parent_path, 'results', 'models', folder)
        out_dir = os.path.join(parent_path, 'results', 'outputs', folder, 'direct')
    else:
        folder = '%d_%s_%s_%s_nl%s_kln%s_dp%s_ep%d_bs%d' % (
            int(time.time()), params['model'], params['loss_fn'],
            os.path.split(args.word_vecs_path)[-1],
            params['num_layers'], os.path.split(args.labels_path)[1],
            params['dropout'], params['num_epochs'], params['batch_size'])

        params['model_dir'] = os.path.join(parent_path, 'results', 'models', folder)
        out_dir = os.path.join(parent_path, 'results', 'outputs', folder)
    os.makedirs(out_dir, exist_ok=True)

    # ------
    # Encode
    estimator = EncodeEstimator(params)

    if args.folder is None:
        if args.test_mode == 2:
            estimator.train_and_eval(
                docs_train, labels_lm_train, labels_train,
                docs_eval, labels_lm_eval, labels_eval, max_step=1)
        else:
            estimator.train_and_eval(
                docs_train, labels_lm_train, labels_train,
                docs_eval, labels_lm_eval, labels_eval)

    estimator.batch_size = 128  # takes less time with large batch size
    doc_vecs, pred_labels = estimator.predict(docs)

    pub_med_ids, _ = read_file(args.documents_path)
    index2word = load(args.index2word_path)

    # save predicted labels to disk
    pred_lab_words = []
    for p_id, lab in zip(pub_med_ids, pred_labels):
        pred_lab = ', '.join([index2word[terms[l]] for l in lab])
        line = str(p_id) + '\t' + pred_lab
        pred_lab_words.append(line)

    save_list(os.path.join(out_dir, 'pred_labels.txt'), pred_lab_words)

    if not args.keep_model_files:
        shutil.rmtree(params['model_dir'], ignore_errors=True)

    # write params to a txt file, except embeddings
    param_dir = os.path.join(out_dir, 'params.txt')
    if 'embeddings' in params:
        del params['embeddings']
    with open(param_dir, 'w') as f:
        f.write(json.dumps(params))
    del params

    print("Finished predicting.")
    if args.no_inference:
        exit(1)

    # ---------
    # Inference
    doc_tfidf_reps = labels
    if len(args.doc_tfidf_reps_path) > 0:
        doc_tfidf_reps = load(args.doc_tfidf_reps_path)

    fused_docs, expanded, top_k_indices = inference.main(
        doc_vecs, doc_tfidf_reps, args.k, args.fuse_doc_type)

    save(os.path.join(out_dir, 'top_k_indices'), top_k_indices)
    if args.keep_model_files:
        np.save(os.path.join(out_dir, 'fused_docs'), fused_docs)
        np.save(os.path.join(out_dir, 'doc_vecs'), doc_vecs)
    del doc_vecs, top_k_indices, fused_docs

    # ----------------------------
    # Save expanded labels to disk
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
    main()
