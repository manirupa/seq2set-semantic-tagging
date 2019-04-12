from collections import defaultdict
import time

from gensim.models import TfidfModel


def doc2bow(document, vocab):
    """ Construct (word, frequency) mapping. """

    counter = defaultdict(int)
    for w in document:
        counter[w] += 1

    return [(vocab[w].index, freq)
            for w, freq in counter.items()
            if w in vocab]


def get_top_n(tfidfs_flattened, n):
    """
    Get the top n terms by tfidf values

    :param tfidfs_flattened: an element in it: (wordID, TF-IDF_value)
    :param n: top n
    :return: top_n, a dict: {wordID: IDF_value}
    """

    # Sort by tfidf value
    # Since one term can have different tf-idf
    # values in different docs, we can't only
    # sort the top n elements of tfidfs_flattened.
    tfidfs_flattened = sorted(
        tfidfs_flattened, key=lambda x: x[1], reverse=True)

    top_n = {}

    for term_tuple in tfidfs_flattened:
        if len(top_n) == n:
            break
        if term_tuple[0] not in top_n:
            top_n[term_tuple[0]] = term_tuple[1]
    return top_n


def get_labels(tfidfs, top_n):
    labels_tuples = []
    tfidf_reps_tuples = []
    for tfidf in tfidfs:
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        term_count = 0
        label = []
        for t_tuple in tfidf:
            if term_count == 3:
                break
            if t_tuple[0] in top_n:
                label.append((t_tuple[0], top_n[t_tuple[0]]))
                term_count += 1
        labels_tuples.append(label)
        tfidf_reps_tuples.append(tfidf[:3])

    # get rid of tfidf value and only keep word id
    labels = [[l[0] for l in label] for label in labels_tuples]
    doc_tfidf_reps = [[l[0] for l in label] for label in tfidf_reps_tuples]
    return labels_tuples, labels, doc_tfidf_reps


def get_terms(labels_tuples):
    terms_tuples = [item for sublist in labels_tuples for item in sublist]
    terms_tuples = list(set(terms_tuples))  # remove duplicates

    # sort based on tfidf values
    terms_tuples = sorted(terms_tuples, key=lambda x: x[1], reverse=True)
    return terms_tuples


def get_tfidf(documents, vocab, n):
    time0 = time.time()
    print("Calculate tfidf values")

    # An element in corpus represents a doc:
    # [(wordID, #occurence), ...],
    # listed in ascending order of wordID.
    corpus = [doc2bow(d, vocab) for d in documents]
    tfidf_model = TfidfModel(corpus=corpus, normalize=False)

    # An element in tfidfs represents a doc:
    # [(wordID, TF-IDF_value), ...],
    # listed in ascending order of wordID.
    tfidfs = [tfidf_model[c] for c in corpus]

    # expand nested list of (wordID, TF-IDF_value) to a 1D list
    tfidfs_flattened = [item for sublist in tfidfs
                        for item in sublist]

    # -----------------
    time1 = time.time()
    print(time1-time0, "s.\nGet top n")
    top_n = get_top_n(tfidfs_flattened, n)

    # -----------------
    time2 = time.time()
    print(time2-time1, "s. \nGet labels")
    labels_tuples, labels, doc_tfidf_reps = get_labels(tfidfs, top_n)

    # ---------
    # Get terms
    terms_tuples = get_terms(labels_tuples)

    wv2terms = {term[0]: i for i, term in enumerate(terms_tuples)}

    # convert ids
    labels = [[wv2terms[lab] for lab in label]
              for label in labels]

    time3 = time.time()
    print(time3 - time2, "s.")

    return labels, terms_tuples, wv2terms, doc_tfidf_reps, tfidf_model
