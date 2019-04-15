import argparse
import os
import re
import time

import numpy as np
from annoy import AnnoyIndex
from nltk.corpus import stopwords

from model_params import get_params
from models.EncodeEstimator import EncodeEstimator
from models.inference import expand_label_v3
from utils import load, save_list, read_file, save

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
        nl,
        root_output_folder,
        folder,
        k):
    params = get_params(model)
    params['embedding_dim'] = init_embed.shape[1]
    params['embeddings'] = init_embed
    params['dropout'] = 0
    params['num_layers'] = nl
    params['num_epochs'] = 1
    params['batch_size'] = 1
    params['term_size'] = 9956
    params['folder'] = model

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
    print(args)
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
                                params['nl'],
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
            params['nl'],
            args.root_output_folder,
            args.folder,
            args.k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Direct.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--txt_docs_path', nargs='?', type=str,
        default='data/cleaned.txt',
        help='30 sum full text file')
    parser.add_argument(
        '--docs_word_indices_path', nargs='?', type=str,
        default='data/docs_word_indices_mc1.pickle',
        help='doc word indices pickle')
    parser.add_argument(
        '--d2v', nargs='?', type=str,
        default='1545725498_doc2vec',
        help='doc2vec folder')
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
        '--index2word_path', nargs='?', type=str,
        default='data/index2word_mc1.pickle',
        help=' ')
    parser.add_argument(
        '--terms_path', nargs='?', type=str,
        default='data/terms.pickle',
        help=' ')
    parser.add_argument(
        '--labels_path', nargs='?', type=str,
        default='data/labels.pickle',
        help=' ')
    parser.add_argument(
        '--fuse_doc_type', nargs='?', type=str,
        default='arithmetic_mean',
        help='fuse doc type')
    parser.add_argument(
        '--root_output_folder', nargs='?', type=str,
        default='/fs/project/PAS0536/seq2set_proj_archive/seq2set_v2_until20190217_outputs',
        help='root folder for putting output inference folders')
    parser.add_argument(
        '--root_folder', nargs='?', type=str,
        default='/fs/project/PAS0536/seq2set_proj_archive/seq2set_v2_until20190217_models',
        help='root folder containing model folders')
    parser.add_argument(
        '--folder', nargs='?', type=str,
        default='1551239956_BiLSTM__SigmoidEncode_word2vec_sg1_s50_w4_m1_n5_i15.'
                'npy_nl4_klnlabels.pickle_dp0.2_ep1_bs32',
        help='model folder')
    parser.add_argument(
        '--folder_names_file_path', nargs='?', type=str,
        default='',
        # default='/fs/project/PAS0536/seq2set_proj_archive/seq2set_v2_until20190217_models/model_names.txt',
        help='if set, then model names file path is used')
    args = parser.parse_args()   
    
    print('folder_names_file_path', args.folder_names_file_path)

    main()
