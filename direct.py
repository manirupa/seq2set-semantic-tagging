import argparse
import os

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords

from model_params import get_params
from models import inference
from models.EncodeEstimator import EncodeEstimator
from utils import load, save_list, save

stop_words = stopwords.words("english")


def main():
    labels = load(args.labels_path)
    terms = load(args.terms_path)

    documents = []
    with open(args.test_doc_path, encoding='utf-8') as input_file:
        for line in input_file:
            # remove '\n' and the last '.'
            line = line.strip('\n. ').lower()
            # remove stop words
            line = [word for word in line.split()
                    if word not in stop_words]
            documents.append(line)

    pred_labels = None

    if args.model == 'doc2vec':
        doc_vecs = []
        model = Doc2Vec.load(os.path.join("results", "models", args.folder, "doc2vec.model"))
        for d in documents:
            doc_vecs.append(model.infer_vector(d))
        out_dir = os.path.join("results", "outputs", args.folder, "direct")
        os.makedirs(out_dir, exist_ok=True)
    else:
        vocab = load(args.vocab_path)
        docs = [[vocab[token] for token in d if token in vocab]
                for d in documents]

        params = get_params(args.model)
        params['embeddings'] = np.load(args.word_vecs_path)
        params['vocab_size'] = params['embeddings'].shape[0]
        params['embeddings_dim'] = params['embeddings'].shape[1]
        params['dropout'] = 0.0
        params['num_epochs'] = 1
        params['batch_size'] = 64
        params['term_size'] = len(terms)
        params['loss_fn'] = args.loss_fn
        params['model'] = args.model
        params['model_dir'] = os.path.join("results", "models", args.folder)

        out_dir = os.path.join("results", "outputs", args.folder, "direct")
        os.makedirs(out_dir, exist_ok=True)

        # get estimator
        estimator = EncodeEstimator(params)

        doc_vecs, pred_labels = estimator.predict(docs)

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
    index2word = load(args.index2word_path)

    if args.model != 'doc2vec':
        pred_lab_words = []
        for lab in pred_labels:
            pred_lab = ', '.join([index2word[terms[l]]
                                  for l in lab])
            line = pred_lab
            pred_lab_words.append(line)

        save_list(out_dir + '/pred_labels.txt', pred_lab_words)

    # convert to word ids
    labels = [[terms[l] for l in lab] for lab in labels]

    if doc_tfidf_reps is None:
        expanded = [[terms[l] for l in lab] for lab in expanded]

    expanded_labels = []
    for l, ex in zip(labels, expanded):
        e_words = ', '.join([index2word[e] for e in ex])
        original = ', '.join([index2word[i] for i in l])
        line = 'ORIGINAL: ' + original + '\tEXPANDED: ' + e_words
        expanded_labels.append(line)

    fname = os.path.split(out_dir)[-1] + '_expanded_labels.txt'
    expanded_labels_dir = os.path.join(out_dir, fname)
    save_list(expanded_labels_dir, expanded_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Direct.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--word_vecs_path', nargs='?', type=str,
        default='data/word2vec_sg1_s50_w4_m1_n5_i15.npy',
        help=' ')
    parser.add_argument(
        '--test_doc_path', nargs='?', type=str,
        default='data/30_sum_full_text.txt',
        help='30 sum full text file')
    parser.add_argument(
        '--doc_tfidf_reps_path', nargs='?', type=str,
        default='data/doc_tfidf_reps_mc1.pickle',
        help='doc_tfidf_reps_path')
    parser.add_argument(
        '-k', nargs='?', type=int,
        default=5, help='top k closest docs of a query doc')
    parser.add_argument(
        '--loss_fn', nargs='?', type=str,
        default='sigmoid',
        help='loss function')
    parser.add_argument(
        '--terms_path', nargs='?', type=str,
        default='data/terms.pickle',
        help=' ')
    parser.add_argument(
        '--index2word_path', nargs='?', type=str,
        default='data/index2word_mc1.pickle',
        help=' ')
    parser.add_argument(
        '--vocab_path', nargs='?', type=str,
        default='data/vocab_mc1.pickle',
        help=' ')
    parser.add_argument(
        '--labels_path', nargs='?', type=str,
        default='data/labels.pickle',
        help=' ')
    parser.add_argument(
        '--model', nargs='?', type=str,
        default='BiLSTM',
        help=' ')
    parser.add_argument(
        '--fuse_doc_type', nargs='?', type=str,
        default='arithmetic_mean',
        help='fuse doc type')
    parser.add_argument(
        '--folder', nargs='?', type=str,
        default='1551239956_BiLSTM__SigmoidEncode_word2vec_sg1_s50_w4_m1_n5_i15.npy_nl4_klnlabels.pickle_dp0.2_ep1_bs32',
        help='model folder')
    args = parser.parse_args()

    main()
