import os
import re
import time

import numpy as np
from annoy import AnnoyIndex
from nltk.corpus import stopwords

from models.EncodeEstimator import EncodeEstimator
from models.inference import expand_label_v3
from utils import load, save_list, read_file, save, get_args

stop_words = stopwords.words("english")


def get_expanded_terms(top_k_indices, labels):
    """

    :param top_k_indices: A collection of all the closest docs.
        An element of top_k_indices: [(doc_id1, similarity1), ..., (doc_id5, similarity5)]
    :param labels: doc_tfidf_reps
    :return:
    """
    expanded = [expand_label_v3(top_k, labels)
                for q, top_k in enumerate(top_k_indices)]

    return expanded


def write_to_file(str_to_print, filename):
    f = open(filename, 'a')
    f.write(str_to_print)
    f.close()


def get_docs_neighbors(doc_vecs, model, k):
    print('Documents vectors shape:', doc_vecs.shape)
    
    size = doc_vecs.shape[0]
    dims = doc_vecs.shape[1]

    t = AnnoyIndex(dims)  # Length of item vector that will be indexed
    for idx in range(size):
        v = doc_vecs[idx].tolist()
        t.add_item(idx, v)
       
    print('added items')
    
    fname = 'doc_vecs_neighbors_%s.ann' % model
    tic = time.time()
    t.build(100)  # 100 trees
    t.save(fname)
    toc = time.time()
    print("Build time:", toc-tic, "seconds")
      
    nn = AnnoyIndex(dims)
    tic = time.time()
    nn.load(fname)  # super fast, will just mmap the file
    toc = time.time()
    print("Load time:", toc-tic, "seconds")
     
    print('Eg. neighbors of 2:', nn.get_nns_by_item(2, k))  # will find the K nearest neighbors
    
    doc_neighbors_list = []
    for idx in range(size):
        idlist = '%s\n' % nn.get_nns_by_item(idx, k)
        doc_neighbors_list.append(idlist)
        # write_to_file(str(idlist), 'testindices.txt')
    return doc_neighbors_list


def inference_from_checkpoint(
        pub_med_ids,
        docs,
        init_embed,
        labels,
        doc_tfidf_reps,
        index2word,
        terms,
        model,
        root_output_folder,
        folder,
        k):
    params = vars(args)
    params['embedding_dim'] = init_embed.shape[1]
    params['embeddings'] = init_embed

    # get estimator
    estimator = EncodeEstimator(params)

    doc_vecs, _ = estimator.predict(docs)
    
    # Get top-k indices for documents
    top_k_indices = get_docs_neighbors(doc_vecs, model, k)
    
    expanded_labels_dir = os.path.join(root_output_folder, folder)
    os.makedirs(expanded_labels_dir, exist_ok=True)

    top_k_indices_path = os.path.join(expanded_labels_dir, 'top_k_indices')
    save(top_k_indices_path, top_k_indices)

    expanded = get_expanded_terms(top_k_indices, doc_tfidf_reps)

    labels = [[terms[l] for l in labs] for labs in labels]

    expanded_labels = []
    for p_id, l, ex in zip(pub_med_ids, labels, expanded):
        e_words = ', '.join([index2word[e] for e in ex])
        original = ', '.join([index2word[i] for i in l])
        line = str(p_id) + '\tORIGINAL: ' + original + '\tEXPANDED: ' + e_words
        expanded_labels.append(line)

    expanded_labels_path = os.path.join(expanded_labels_dir, folder + '_expanded_labels_top1.txt')
    save_list(expanded_labels_path, expanded_labels)


def get_ind(_list, part):
    for x in _list:
        if x.endswith(part):
            return _list.index(x)


def parse_folder_name(folder_name):
    _list = folder_name.split('_')
    word_vec_path = 'data/%s' % '_'.join(_list[get_ind(_list, 'Encode') + 1:get_ind(_list, '.npy') + 1])
    nl_part = _list[get_ind(_list, '.npy') + 1]
    nl = int(re.findall(r'\d+', nl_part)[0])
    model_name = _list[1]
    return {'word_vec_path': word_vec_path, 'nl': nl, 'model_name': model_name}


def main():
    doc_tfidf_reps = load(args.doc_tfidf_reps_path)
    labels = load(args.labels_path)
    index2word = load(args.index2word_path)
    terms = load(args.terms_path)
    pub_med_ids, documents = read_file(args.txt_docs_path)
    
    docs = load(args.docs_word_indices_path)
    
    # Batch
    if args.folder_names_file_path: 
        model_list = [x.strip() for x in open(args.folder_names_file_path).readlines()] 
        for model_folder in model_list:
            print("Model: %s" % model_folder)
            # model_path = os.path.join(args.root_output_folder, model_name)
            params = parse_folder_name(model_folder)
            init_embed = np.load(params['word_vec_path'])
            inference_from_checkpoint(
                                pub_med_ids,
                                docs,
                                init_embed,
                                labels,
                                doc_tfidf_reps,
                                index2word,
                                terms,
                                params['model_name'],
                                args.root_output_folder,
                                model_folder,
                                args.k)
 
    else:
        params = parse_folder_name(args.folder)
        init_embed = np.load(params['word_vec_path'])
        
        # Single
        inference_from_checkpoint(
            pub_med_ids,
            docs,
            init_embed,
            labels,
            doc_tfidf_reps,
            index2word,
            terms,
            params['model_name'],
            args.root_output_folder,
            args.folder,
            args.k)


if __name__ == '__main__':
    args = get_args()
    main()
