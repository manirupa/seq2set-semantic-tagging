import os
import time

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from models import inference
from utils import load, save, save_list, read_file, get_args


def main():
    # ---------
    # load data
    args = get_args()
    pub_med_ids, documents = read_file(args.documents_path)
    labels = load(args.labels_path)
    terms = load(args.terms_path)

    if args.test_mode != 0:
        documents = documents[:100]
        labels = labels[:100]

    # ------
    # Encode
    folder = str(int(time.time())) + '_doc2vec'
    model_dir = 'results/models/' + folder
    os.makedirs(model_dir, exist_ok=True)
    out_dir = 'results/outputs/' + folder
    os.makedirs(out_dir, exist_ok=True)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    model = Doc2Vec(documents, vector_size=300, window=10, min_count=5, workers=-1, epochs=20)
    model.save(model_dir + '/doc2vec.model')
    doc_vecs = model.docvecs.vectors_docs

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

    # ---------
    # save data

    # convert to word ids
    labels = [[terms[l] for l in lab] for lab in labels]

    if len(args.doc_tfidf_reps_path) == 0:
        expanded = [[terms[l] for l in lab] for lab in expanded]

    index2word = load(args.index2word_path)
    expanded_labels = []
    for p_id, l, ex in zip(pub_med_ids, labels, expanded):
        e_words = ', '.join([index2word[e] for e in ex])
        original = ', '.join([index2word[i] for i in l])
        line = str(p_id) + '\tORIGINAL: ' + original + '\tEXPANDED: ' + e_words
        expanded_labels.append(line)

    expanded_labels_dir = out_dir + '/' + out_dir.split('/')[-1] + '_expanded_labels.txt'
    save_list(expanded_labels_dir, expanded_labels)


if __name__ == '__main__':
    main()
