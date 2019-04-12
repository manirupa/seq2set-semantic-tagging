""" Run On OSC """

import numpy as np


def get_cosine_similarity(mat):
    # base similarity matrix (all dot products)
    similarity = np.dot(mat, mat.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (element-wise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine


def k_largest(arr, k):
    doc_ids = np.argpartition(arr, -k)[-k:]
    similarities = arr[doc_ids]
    return np.dstack((doc_ids, similarities))[0]


def expand_label(top_k, q, labels):
    """
    Expand the labels of a query doc by selecting 1 term from each of the closest docs of the query doc.
    :param top_k: the ids of the top k closest docs of the query doc,
        shape: [(doc_id, similarity between doc[doc_id] and doc[q]), ...]
    :param q: query doc id
    :param labels: the set of candidate labels for expansion
    :return: expanded term ids of the query doc
    """

    term_ids = []
    for i, (doc_id, _) in enumerate(top_k):
        for lab in labels[doc_id]:
            if lab not in term_ids and lab not in labels[q]:
                term_ids.append(lab)
                break
    return term_ids


def expand_label_v2(top_k, labels):
    """
    Expand the labels of a query doc by selecting 3 terms from all k closest docs of a query doc
    :param top_k: the ids of the top k closest docs of the query doc,
        shape: [(doc_id, similarity between doc[doc_id] and doc[q]), ...]
    :param labels: the set of candidate labels for expansion
    :return: expanded term ids of the query doc
    """

    # all the labels of the k closest docs, shape: [word_id0, ...]
    term_ids = [lid for doc_tuple in top_k for lid in labels[int(doc_tuple[0])]]

    # remove duplicates and convert to np array
    term_ids = np.array(list(set(term_ids)))
    if term_ids.size <= 3:
        return term_ids

    term_tfidfs = np.array([r for r in term_ids])
    arg_part = np.argpartition(term_tfidfs, -3)[-3:]
    return term_ids[arg_part]


def expand_label_v3(top_k, labels):
    """
    Expand the labels of a query doc by selecting 3 terms from the closest docs of a query doc.
    :param top_k: the ids of the top k closest docs of the query doc,
        shape: [(doc_id, similarity between doc[doc_id] and doc[q]), ...]
    :param labels: the set of candidate labels for expansion
    :return: expanded term ids of the query doc
    """

    closest_doc_id = top_k[0][0]
    term_ids = labels[closest_doc_id]
    return term_ids


def fuse_docs_v2(to_be_fused):
    """
    Fuse the doc representation of several docs together
    (geometric mean)
    """

    ret = np.ones(to_be_fused.shape[1])
    for xi in to_be_fused:
        ret = ret * xi

    return np.power(ret, 1/to_be_fused.shape[0])


def fuse_docs_v1(to_be_fused):
    """
    Fuse the doc representation of several docs together
    (arithmetic mean)
    """

    return np.mean(to_be_fused, axis=0)


def main(doc_vecs, labels, k, fuse_doc_type):
    if fuse_doc_type == 'arithmetic_mean':
        fuse_docs_fn = fuse_docs_v1
    elif fuse_doc_type == 'arithmetic_mean':
        fuse_docs_fn = fuse_docs_v2
    else:
        raise ValueError('Wrong fuse_doc_type!')

    # ------------------
    # Calculate distance
    cosine_similarity = get_cosine_similarity(doc_vecs)

    # set diagonal elements to -1 to avoid picking them in the next step
    np.fill_diagonal(cosine_similarity, -1)

    # ----------------------------------------------------
    # Find the top k closest doc of each doc, save to disk
    #
    # An element of top_k_indices:
    # [(doc_id1, similarity1), ..., (doc_id5, similarity5)]
    top_k_indices = np.apply_along_axis(k_largest, 1, cosine_similarity, k)
    top_k_indices = [[(int(id_pair[0]), id_pair[1])
                      for id_pair in top_k_id] for top_k_id in top_k_indices]

    del cosine_similarity

    # ------------------
    # Get expanded terms
    expanded = [expand_label(top_k, q, labels)
                for q, top_k in enumerate(top_k_indices)]

    # ---------
    # Fuse docs
    fused_docs = np.zeros(doc_vecs.shape)
    for i, top_k in enumerate(top_k_indices):
        doc_ids = [int(t[0]) for t in top_k]
        to_be_fused = doc_vecs[doc_ids]
        fused_docs[i] = fuse_docs_fn(to_be_fused)

    return fused_docs, expanded, top_k_indices
