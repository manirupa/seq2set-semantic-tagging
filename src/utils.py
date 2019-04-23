import pickle
import argparse


def read_file(filename):
    with open(filename, encoding='utf-8') as input_file:
        # each document should be a list of words
        documents = []
        pub_med_ids = []

        for line in input_file:
            tmp = line.split('\t')
            documents.append(tmp[1].split())
            pub_med_ids.append(tmp[0])
    return pub_med_ids, documents


def save_list(file_name, my_list):
    with open(file_name, mode='w', encoding='utf-8') as f:
        for line in my_list:
            f.write(line if line[-1] == '\n' else line+'\n')


def save(file_name, save_object):
    with open(file_name, 'wb') as fp:
        pickle.dump(
            save_object, fp,
            protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def remove_samples_with_empty_labels(docs, labels):
    """ remove docs whose label set is empty, truncates docs """

    docs_new = []
    labels_new = []

    for doc, label in zip(docs, labels):
        if len(label) != 0:
            docs_new.append(doc)
            labels_new.append(label)

    return docs_new, labels_new


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='Transformer',
        help='options: DAN, LSTM, BiLSTM, BiLSTMATT, Transformer, doc2vec')
    parser.add_argument(
        '--loss_fn', default='lm',
        help='options: softmax_uniform, softmax_skewed_labels, lm, sigmoid, sigmoid_with_constraint')
    parser.add_argument(
        '--test_mode', default=0, type=int, help='if 0, normal.'
        'if 1, inference will only calculate the cosine'
        'similarities of the first 100 docs.'
        'if 2, inference will only calculate the cosine'
        'similarities of the first 100 docs and Encoder'
        'will only train for 1 step.')
    parser.add_argument('--fuse_doc_type', default='arithmetic_mean', help='options: arithmetic_mean, geometric_mean')
    parser.add_argument('--keep_model_files', action='store_true')
    parser.add_argument('--no_inference', action='store_true')
    # input file paths
    parser.add_argument('--word_vecs_path', default='data/word2vec_sg1_s50_w4_m1_n5_i15.npy')
    parser.add_argument('--docs_path', default='data/docs_word_indices_mc1.pickle')
    parser.add_argument('--labels_path', default='data/labels.pickle')
    parser.add_argument('--doc_tfidf_reps_path', default='data/doc_tfidf_reps_mc1.pickle')
    parser.add_argument('--index2word_path', default='data/index2word_mc1.pickle')
    parser.add_argument('--terms_path', default='data/terms.pickle')
    parser.add_argument('--documents_path', default='data/cleaned.txt')  # data/cleaned_phrase_embedded.txt
    parser.add_argument('--embedded_sentences', default='data/elmo_50_sentences.hdf5')  # elmo
    parser.add_argument('--folder')

    parser.add_argument('--k', default=5, type=int, help='top k closest docs of a query doc')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of LM loss')
    # adam optimizer params
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-9)

    parser.add_argument('--doc_vec_length', type=int, default=50)
    # the number of distinct labels
    parser.add_argument('--term_size', type=int, default=9956)

    # Dimensions of the dense layers, where the last dim equals term_size.
    # And there must exists one layer whose dim == doc_vec_length
    parser.add_argument('--mlp_layer_dims', nargs='+', type=int, default=[50, 1000, 2500, 5000, 9956])
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attention_size', type=int, default=200)  # BiLSTMATT
    parser.add_argument('--num_heads', type=int, default=5)  # Transformer
    # non-model params
    parser.add_argument('--max_length', type=int, default=256)  # used for truncating
    parser.add_argument('--padding_length', type=int, default=256)  # used for padding
    args = parser.parse_args()
    return args
