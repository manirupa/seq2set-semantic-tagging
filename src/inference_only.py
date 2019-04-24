import os
import json
import sys

import numpy as np

from models import inference
from utils import load, save, save_list, read_file, DotDict


def inference_only(param_path):
    # -----
    # load data
    with open(param_path) as f:
        args = json.load(f)

    args = DotDict(args)
    out_dir = args.model_dir.replace('model', 'output')
    doc_vecs_path = os.path.join(out_dir, 'doc_vecs.npy')

    pub_med_ids, _ = read_file(args.documents_path)
    labels = load(args.labels_path)
    index2word = load(args.index2word_path)
    terms = load(args.terms_path)
    doc_vecs = np.load(doc_vecs_path)

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


def main(fname):
    param_path_list = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            param_path_list.append(line)

    for param_path in param_path_list:
        inference_only(param_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("only 1 argument allowed")
    main(str(sys.argv[1]))
