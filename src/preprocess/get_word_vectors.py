import time
import os

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText as FT_gensim


def get_w2v_model(documents, settings):
    return Word2Vec(
        documents, min_count=settings['min_count'],
        size=settings['size'], window=settings['window'],
        workers=40, sg=settings['sg'],
        negative=settings['negative'], iter=settings['iter'])


def get_ft_model(documents, settings):
    model = FT_gensim(
        min_count=settings['min_count'],
        size=settings['size'], window=settings['window'],
        workers=40, sg=settings['sg'],
        negative=settings['negative'], iter=settings['iter'])

    model.build_vocab(documents)

    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
    return model


def get_wv(documents, settings, model_name):
    time0 = time.time()
    print("Get wv")

    # Get dir to save the model
    s_w = '_sg{}_s{}_w{}_m{}_n{}_i{}'.format(
        settings['sg'], settings['size'], settings['window'],
        settings['min_count'], settings['negative'], settings['iter'])
    model_dir = 'results/models/' + model_name + s_w
    wv_dir = model_dir + '/wv'

    if os.path.isdir(model_dir):
        print("This wv model has already been trained.")
        return wv_dir

    if model_name == "fast_text":
        model = get_ft_model(documents, settings)
    elif model_name == "word2vec":
        model = get_w2v_model(documents, settings)
    else:
        raise ValueError("Invalid model name. Options: fast_text or word2vec")

    keyed_vectors = model.wv

    # save model
    os.makedirs(model_dir, exist_ok=True)
    keyed_vectors.save(wv_dir)

    # save word_vectors
    np.save('data/' + model_name + s_w,
            np.append(  # append <PAD> to word vectors
                keyed_vectors.vectors,
                np.zeros((1, keyed_vectors.vectors.shape[1])),
                axis=0)
            )

    # append padding value <PAD> to index2word
    index2word = keyed_vectors.index2word
    index2word.append('<PAD>')

    print(time.time()-time0, "s.")
    return wv_dir
