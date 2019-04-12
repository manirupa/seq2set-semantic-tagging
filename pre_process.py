import argparse
import os

from nltk.corpus import stopwords
from gensim.models import KeyedVectors

from preprocess.get_tfidf import get_tfidf
from preprocess.get_word_vectors import get_wv
from utils import save_list, save, read_file
from model_params import settings

stop_words = stopwords.words("english")


def main(input_dir, model_name, n):
    pub_med_ids, documents = read_file(input_dir)

    # ---
    # w2v
    wv_dir = get_wv(documents, settings, model_name)

    min_count = str(settings['min_count'])

    keyed_vectors = KeyedVectors.load(wv_dir)
    vocab = keyed_vectors.vocab
    index2word = keyed_vectors.index2word

    save_list('data/index2word_mc' + min_count + '.txt', index2word)
    save('data/index2word_mc' + min_count + '.pickle', index2word)

    # -----
    # tfidf
    tfidf_model_dir = 'results/models/tfidf_model_mc' + min_count
    if os.path.isfile(tfidf_model_dir):
        print("This tfidf model has already been trained.")
        return
    labels, terms_tuples, wv2terms, doc_tfidf_reps, tfidf_model = get_tfidf(
        documents, vocab, n)

    tfidf_model.save(tfidf_model_dir)

    # ------------
    # save to disk

    # convert to word ids
    docs = [[vocab[token].index for token in d if token in vocab]
            for d in documents]

    terms_txt = ['{}\t{}'.format(index2word[t[0]], t[1])
                 for t in terms_tuples]

    # get rid of tfidf value and only keep word id
    terms = [t[0] for t in terms_tuples]

    labels_txt = ['{}\t{}'.format(pub_med_id,
                                  ', '.join([index2word[terms[l]]
                                             for l in lab]))
                  for pub_med_id, lab in zip(pub_med_ids, labels)]

    doc_tfidf_reps_txt = ['{}\t{}'.format(pub_med_id,
                                          ', '.join([index2word[l]
                                                     for l in lab]))
                          for pub_med_id, lab in zip(pub_med_ids, doc_tfidf_reps)]

    save_list('data/terms_mc' + min_count + '.txt', terms_txt)
    save_list('data/labels_mc' + min_count + '.txt', labels_txt)
    save_list('data/doc_tfidf_reps_mc' + min_count + '.txt', doc_tfidf_reps_txt)

    save('data/docs_word_indices_mc' + min_count + '.pickle', docs)
    save('data/labels_mc' + min_count + '.pickle', labels)
    save('data/doc_tfidf_reps_mc' + min_count + '.pickle', doc_tfidf_reps)
    save('data/wv2terms_mc' + min_count + '.pickle', wv2terms)
    save('data/terms_mc' + min_count + '.pickle', terms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-processing...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', nargs='?', type=str,
        default='data/cleaned.txt',
        help='input source data file, '
             'options: data/cleaned.txt or cleaned_phrase_embedded.txt')
    parser.add_argument(
        '-m', nargs='?', type=str,
        default='fast_text',
        help='model to train word vectors, options: fast_text or word2vec')
    parser.add_argument(
        '-n', nargs='?', type=int,
        default=10000, help='top n tfidf')
    args = parser.parse_args()

    main(args.i, args.m, args.n)
