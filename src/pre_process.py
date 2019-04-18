import argparse
import os

from nltk.corpus import stopwords
from gensim.models import KeyedVectors

from preprocess.get_tfidf import get_tfidf
from preprocess.get_word_vectors import get_wv
from utils import save_list, save, read_file

stop_words = stopwords.words("english")


def main():
    pub_med_ids, documents = read_file(args.i)
    settings = vars(args)

    # ---
    # w2v
    wv_dir = get_wv(documents, settings, args.m)

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
        documents, vocab, args.n)

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', default='data/cleaned.txt',
        help='input source data file, '
             'options: data/cleaned.txt or cleaned_phrase_embedded.txt')
    parser.add_argument(
        '-m', default='fast_text',
        help='model to train word vectors, options: fast_text or word2vec')
    parser.add_argument('-n', type=int, default=10000, help='top n tfidf')
    parser.add_argument(
        '--sg', type=int, default=1,
        help='Training algorithm: 1 for skip-gram; otherwise CBOW.')
    parser.add_argument('--size', type=int, default=50, help='Dimensionality of the word vectors.')
    parser.add_argument(
        '--window', type=int, default=4,
        help='Maximum distance between the current and predicted word within a sentence.')
    parser.add_argument(
        '--min_count', type=int, default=1,
        help='Ignores all words with total frequency lower than this.')
    parser.add_argument(
        '--negative', type=int, default=5,
        help='If > 0, negative sampling will be used, the int'
             'for negative specifies how many "noise words"'
             'should be drawn (usually between 5-20).'
             'If set to 0, no negative sampling is used.')
    parser.add_argument('--iter', type=int, default=15, help='Number of epochs over the corpus.')
    args = parser.parse_args()

    main()
