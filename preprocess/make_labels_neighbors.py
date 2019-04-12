import argparse

import numpy as np

from utils import load, save, save_list
from models.inference import k_largest, expand_label


def jaccard_similarity(list1, list2):
    len1 = len(list1)
    len2 = len(list2)

    if len1 == 0:
        return 1.0 if len2 == 0 else 0.0
    elif len2 == 0:
        return 0.0

    s1 = set(list1)
    s2 = set(list2)

    intersection_len = len(s1.intersection(s2))

    if intersection_len == 0:
        return 0
    return intersection_len / (len1 + len2 - intersection_len)


def get_similarity_matrix(labels):
    n = len(labels)
    xs, ys = np.triu_indices(n, k=1)
    labels = np.array(labels)

    similarities = np.fromiter(
        (jaccard_similarity(xi[0], xi[1])
         for xi in np.dstack((labels[xs], labels[ys]))[0]),
        np.float, count=int(n*(n-1)/2))

    similarity_matrix = np.zeros((n, n))
    similarity_matrix[xs, ys] = similarities
    similarity_matrix[ys, xs] = similarities
    similarity_matrix[np.diag_indices(n)] = -1.0
    return similarity_matrix


def main(labels, n):
    top_k_indices = np.apply_along_axis(
        k_largest, 1,
        get_similarity_matrix(labels),
        n)

    top_k_indices = [[(int(doc_id), similarity) for (doc_id, similarity) in top_k_id]
                     for top_k_id in top_k_indices]

    save('data/tfidf_top{}indices.pickle'.format(n), top_k_indices)

    return [expand_label(top_k, q, labels)
            for q, top_k in enumerate(top_k_indices)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make labels neighbors.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-l', nargs='?', type=str,
        default='../data/labels_mc1.pickle',
        help='input labels pickle file')
    parser.add_argument(
        '-i', nargs='?', type=str,
        default='../data/index2word_mc1.pickle',
        help='word id to actual word mapping file')
    parser.add_argument(
        '-t', nargs='?', type=str,
        default='../data/terms_mc1.pickle',
        help='term id to actual word id mapping file')
    parser.add_argument(
        '-n', nargs='?', type=int,
        default=3, help='number of neighbors')
    parser.add_argument(
        '--test', help='Test on less data',
        action='store_true')
    args = parser.parse_args()

    initial_labels = load(args.l)
    index2word = load(args.i)
    terms = load(args.t)

    if args.test:
        initial_labels = initial_labels[:1000]

    extra_labels = main(initial_labels, args.n)

    expanded_labels = []
    expanded_labels_txt = []
    for label, ex in zip(initial_labels, extra_labels):
        expanded_labels.append(label + ex)

        original = ', '.join([index2word[terms[l]] for l in label])
        e_words = ', '.join([index2word[terms[e]] for e in ex])
        line = 'ORIGINAL: ' + original + '\tEXPANDED: ' + e_words
        expanded_labels_txt.append(line)

    save_list('../data/' + str(args.n) + '_label_neighbors.txt', expanded_labels_txt)
    save('../data/' + str(args.n) + '_label_neighbors.pickle', expanded_labels)
