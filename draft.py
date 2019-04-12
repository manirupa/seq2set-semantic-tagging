import os
import pprint
import time
import re

from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
import pandas as pd

from model_params import get_params
from utils import load, save_list


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)
pp = pprint.PrettyPrinter()


def make_phrases_test():
    from textblob import TextBlob
    from preprocess import make_phrases

    doc_name = 'cleaned'
    with open('data/' + doc_name + '.txt') as old:
        old_lines = old.readlines()

    with open('data/' + doc_name + '_phrase_embedded.txt') as new:
        new_lines = new.readlines()

    if len(old_lines) == len(new_lines):
        print('They both have {} docs.'.format(len(old_lines)))
    else:
        print('Docs numbers don\'t match.')

    not_match_count = 0
    for i, old_line in enumerate(old_lines):
        same_length = len(old_line) != len(new_lines[i])
        new_lines_text = new_lines[i].split('\t')[1]
        old_lines_text = old_lines.split('\t')[1]
        if same_length and new_lines_text.split()[-1] != old_lines_text.split()[-1]:
            if new_lines_text.split('_')[-1] != old_lines_text.split()[-1]:
                not_match_count += 1
                print(i, new_lines_text.split()[-1])

    print('\nNumber of docs not matched: ' + repr(not_match_count))

    line = old_lines[62059].split('\t')[1]
    blob = TextBlob(line)

    phrases_joined = []

    for noun_phrase in blob.noun_phrases:
        # only include NPs with length between 2 and 3
        if 1 <= noun_phrase.count(' ') <= 2:
            phrases_joined.append(noun_phrase)

    result = make_phrases.embed_phrases(line, phrases_joined)
    print(result)


def labels_test():
    from gensim.models import KeyedVectors
    from preprocess.get_tfidf import get_tfidf
    from utils import read_file

    # input_file_name = 'data/cleaned_phrase_embedded.txt'
    input_file_name = 'data/cleaned.txt'

    _, documents = read_file(input_file_name)

    vocab = KeyedVectors.load('results/models/w2v_s300_w5/wv').vocab

    with open('no_labels', 'w') as f:
        for n in [12000, 10000, 8000, 6000, 5000]:
            labels, _, _, _ = get_tfidf(documents, vocab, n)
            less_than_3_labels = []
            no_label = []
            for i, label in enumerate(labels):
                if len(label) < 3:
                    less_than_3_labels.append(i)
                    if len(label) == 0:
                        no_label.append(i)

            len_labels = len(labels)

            print("n: {}".format(n))
            print("less than 3 labels: {}%".format(round(len(less_than_3_labels) / len_labels * 100, 2)))
            print("no labels: {}%".format(round(len(no_label) / len_labels * 100, 2)))
            print('------------------')

            f.write("n: {}\n".format(n))
            f.write("less than 3 labels: {}%\n".format(round(len(less_than_3_labels) / len_labels * 100, 2)))
            f.write("no labels: {}%\n".format(round(len(no_label) / len_labels * 100, 2)))
            f.write('------------------\n\n')


def dataset_test():
    from models.EncodeEstimator import EncodeEstimator
    from encode_and_inference import remove_samples_with_empty_labels

    model = "GRU"
    loss_fn = "sigmoid"
    num_epochs = 1
    batch_size = 32
    dropout = 0.5
    term_size = 9956
    docs_path = "data/docs_word_indices_mc1.pickle"
    labels_path = "data/labels.pickle"
    word_vecs_path = "data/word2vec_sg1_s50_w4_m1_n5_i15.npy"
    terms_path = "data/terms.pickle"

    # load data
    docs = load(docs_path)
    init_embed = np.load(word_vecs_path)
    labels = load(labels_path)
    terms = load(terms_path)

    labels_lm = [[terms[l] for l in lab] for lab in labels]

    # load params
    params = get_params(model)
    params['vocab_size'] = init_embed.shape[0]
    params['embedding_dim'] = init_embed.shape[1]
    params['mlp_layer_dims'] += [term_size]
    params['embeddings'] = init_embed
    params['dropout'] = dropout
    params['kln'] = "qnmd"  # kln: k-label-neighbor
    params['word_vecs_path'] = word_vecs_path
    params['num_epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['term_size'] = term_size
    params['loss_fn'] = loss_fn

    folder = '%d_%s_%s_%s_nl%s_kln%s_dp%s_ep%d_bs%d' % (
        int(time.time()), params['model'], params['loss_fn'],
        os.path.split(params['word_vecs_path'])[-1], params['num_layers'],
        params['kln'], params['dropout'], params['num_epochs'],
        params['batch_size'])

    params['model_dir'] = os.path.join("results", "models", folder)

    estimator = EncodeEstimator(params)

    estimator.n = len(labels)

    encoder = estimator.encoder

    ds = estimator.train_input_fn(docs, labels_lm, labels)

    iterator = ds.make_initializable_iterator()
    features, labels = iterator.get_next()

    vocab_size = params['embeddings'].shape[0]
    embedding_dim = params['embeddings'].shape[1]

    w_embed = tf.get_variable(
        'w_embed',
        shape=[vocab_size, embedding_dim],
        initializer=tf.constant_initializer(
            params['embeddings'], verify_shape=True))

    del params['embeddings']

    with tf.name_scope("embedding"):
        x = tf.nn.embedding_lookup(w_embed, features['sentence'], name='embed')

    with tf.name_scope("lstm"):
        # reshape embed to shape [sequence_len, batch_size, embedding_dim]
        # x = tf.transpose(x, [1, 0, 2])
        gru = tf.keras.layers.GRU(params['hidden_size'], return_sequences=False, return_state=True)
        hhhhhhhhhhh = gru(x)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        hhhhhhhhhhh1 = sess.run(hhhhhhhhhhh)
        print(1)


def dataset_test3():
    from models.EncodeEstimatorElmo import EncodeEstimator
    import h5py

    model = "Transformer"
    loss_fn = "sigmoid"
    num_epochs = 2
    batch_size = 2
    dropout = 0.5
    term_size = 9956
    labels_path = "data/labels_mc1.pickle"
    word_vecs_path = "hahahaha"
    docs_path = 'data/elmo_100_sentences.hdf5'

    # load data
    labels = load(labels_path)

    def docs_gen(_f):
        """

        :param _f: embedded_sentences_file
        :return:
        """
        f_keys = sorted(list(_f.keys()), key=int)

        for k in f_keys:

            # todo debug
            if k == '100':
                break

            embedded_sentence = np.array(_f.get(k))
            yield embedded_sentence, len(embedded_sentence)

    # load params
    params = get_params(model)
    params['word_vecs_path'] = word_vecs_path
    params['mlp_layer_dims'] += [term_size]
    params['dropout'] = dropout
    params['kln'] = "qnmd"  # kln: k-label-neighbor
    params['num_epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['term_size'] = term_size
    params['embedding_dim'] = 100
    params['data_size'] = 100
    params['folder'] = None
    params['loss_fn'] = loss_fn
    params['is_hub'] = False

    encode_estimator = EncodeEstimator(params)

    with h5py.File(docs_path, 'r') as f:
        def sen_gen():
            for i in docs_gen(f):
                yield i[0]

        def len_gen():
            for i in docs_gen(f):
                yield i[1]

        # ==== Check training in estimator =========
        # encode_estimator.train(sen_gen, len_gen, labels, 1)

        # ==== Check model functions without estimator =========
        encode_estimator.n = len(labels)
        ds = encode_estimator.train_input_fn(sen_gen, len_gen, labels)
        iterator = ds.make_one_shot_iterator()
        features, labels = iterator.get_next()

        x, enc_self_att_bias = encode_estimator.encoder.model(features['sentence'], features['length'])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x0, enc_self_att_bias0 = sess.run([x, enc_self_att_bias])
            print(x0, enc_self_att_bias0)


def keywords_test():
    with open('data/misc/PMCID+keywords.txt', encoding='utf-8') as f:
        ids = []
        tags = []
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) > 1:
                ids.append(line[0])
                tags.append(line[1])
            # else:
            #     ids.append(line[0])
            #     tags.append('')
        ids = ids[1:]
        tags = tags[1:]

    pp.pprint(len(ids))
    pp.pprint(len(tags))

    pp.pprint(tags[:10])

    tags_unique = []
    for tag in tags:
        tags_unique.extend(tag.split('|'))

    tags_unique = list(set(tags_unique))

    pp.pprint(len(tags_unique))


def one_hot_test():
    # TODO
    x = tf.constant([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.]])
    x = tf.map_fn(lambda inp: tf.reshape(tf.where(tf.equal(inp, 1.)), [-1]),
                  x,
                  dtype=tf.int64)

    # x = tf.constant([0., 1., 1.])
    # x = tf.pad(tf.reshape(tf.where(tf.equal(x, 1.)), [-1]), paddings=[])

    with tf.Session() as sess:
        print(sess.run(x))


def tf_records():
    def make_example(pub_med_id, sequence, labels):
        # The object we return
        example = tf.train.SequenceExample()

        # A non-sequential feature of our example
        sequence_length = len(sequence)
        example.context.feature["length"].int64_list.value.append(sequence_length)
        example.context.feature["id"].int64_list.value.append(pub_med_id)

        # Feature lists for the two sequential features of our example
        fl_tokens = example.feature_lists.feature_list["tokens"]
        for token in sequence:
            fl_tokens.feature.add().int64_list.value.append(token)

        fl_labels = example.feature_lists.feature_list["label"]
        for label in labels:
            fl_labels.feature.add().int64_list.value.append(label)

        return example

    def parse(example):
        # Define how to parse the example
        context_features = {
            "id": tf.FixedLenFeature([], dtype=tf.int64),
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "label": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        # Parse the example
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return context_parsed, sequence_parsed

    def main():
        sequences = load("data/docs_word_indices")
        label_sequences = load("data/labels")
        pub_med_ids = load("data/pub_med_ids")
        lengths = [len(s) for s in sequences]

        df = pd.DataFrame.from_dict({
            "id": pub_med_ids,
            "length": lengths,
            "tokens": sequences,
            "label": label_sequences
        })

        df = df.sort_values(by=['length'])

        pp.pprint(df.head())

        # Write all examples into a TFRecords file
        with tf.python_io.TFRecordWriter("t.tfrecords") as writer:
            for pub_med_id, sequence, label_sequence in zip(pub_med_ids, sequences, label_sequences):
                example = make_example(pub_med_id, sequence, label_sequence).SerializeToString()
                writer.write(example)

    main()
    dataset = tf.data.TFRecordDataset(['t.tfrecords'])
    dataset = dataset.map(parse, num_parallel_calls=5)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    training_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:
        sess.run(training_init_op)
        x = sess.run(next_element)
        print(x)


def inference_test():
    from models.inference import main
    doc_vecs = np.load('results/outputs/doc2vec/doc_vecs.npy')
    labels = load('data/labels')
    main(doc_vecs[:100], labels[:100], 3, 'arithmetic_mean')


def save_load_variable_test():
    tf.reset_default_graph()

    # Create some variables.
    x = tf.get_variable("dense/kernel", shape=[300, 500])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "results/models/1545453462_DAN_SigmoidEncode_ep1/model.ckpt-1")
        print("Model restored.")
        # Check the values of the variables
        print("x : %s" % x.eval())


def get_tfidf_test():
    from pre_process import read_file
    from preprocess.get_tfidf import get_tfidf
    from gensim.models import KeyedVectors

    input_dir = 'data/cleaned.txt'
    keyed_vectors = KeyedVectors.load('results/models/no_phrase/word2vec_sg1_s50_w4_m1_n5_i15/wv')
    # keyed_vectors = KeyedVectors.load('results/models/word2vec_sg1_s50_w4_m1_n5_i15/wv')
    vocab = keyed_vectors.vocab
    # index2word = keyed_vectors.index2word
    n = 10000

    pub_med_ids, documents = read_file(input_dir)

    labels, terms_tuples, wv2terms, doc_tfidf_reps, _ = get_tfidf(
        documents, vocab, n)

    # labels_txt = ['{}\t{}'.format(pub_med_id,
    #                               ', '.join([index2word[terms[l]]
    #                                          for l in lab]))
    #               for pub_med_id, lab in zip(pub_med_ids, labels)]
    #
    # doc_tfidf_reps_txt = ['{}\t{}'.format(pub_med_id,
    #                                       ', '.join([index2word[l]
    #                                                  for l in lab]))
    #                       for pub_med_id, lab in zip(pub_med_ids, doc_tfidf_reps)]
    print(0)


def vocab_test():
    from gensim.models import KeyedVectors

    keyed_vectors = KeyedVectors.load('results/models/word2vec_sg1_s300_w5_m1_n5_i15/wv')
    # keyed_vectors1 = KeyedVectors.load('results/models/word2vec_sg1_s200_w7_m1_n5_i15/wv')
    keyed_vectors1 = KeyedVectors.load('results/models/word2vec_sg1_s100_w5_m1_n5_i15/wv')
    # keyed_vectors1 = KeyedVectors.load('results/models/word2vec_sg1_s50_w4_m1_n5_i15/wv')

    index2word = keyed_vectors.index2word
    index2word1 = keyed_vectors1.index2word

    print(index2word == index2word1)

    vocab = keyed_vectors.vocab
    vocab1 = keyed_vectors1.vocab

    for key in vocab:
        if vocab[key].count != vocab1[key].count:
            print(key, "\tcount:\t", vocab[key].count, vocab1[key].count)
        if vocab[key].index != vocab1[key].index:
            print(key, "\tindex:\t", vocab[key].index, vocab1[key].index)


"""
from preprocess import make_phrases

make_docs()
clean_data('raw_data/raw_docs_62k.txt')

After clean_data() Manually deleted the following documents:
    videovideo
    figure numbers_decimals
    supplemental digital content is available in the text
    article first published online numbers_decimals ...
    numbers_decimals cancer research campaign
    british journal cancer numbers_decimals . numbers_decimals cancer research uk
    earl brewer discusses journey pediatric rheumatology numbers_decimals retirement numbers_decimals three parts
    the abstract is available clinical pancreatic_disorder acute pancreatitis ...

make_phrases.main('data/cleaned.txt')
"""


def make_docs():
    start = time.time()
    with open('raw_data/docs.txt') as docs62k:
        pcm_ids = [line.split('\t')[0] for line in docs62k]

    def find_line(line):
        if line.split('\t')[0] in pcm_ids:
            return line

    with open('raw_data/sampleval_and_SE_good_docs.txt') as raw:
        docs62k = Parallel(n_jobs=8)(delayed(find_line)(line) for line in raw)

    save_list('raw_data/raw_docs_62k.txt', docs62k)

    print("Finished making docs. Time: {}".format(time.time() - start))


def clean_line(line):
    # if ' une ' in line and ' est ' in line:
    #     return None

    line = re.sub(r'[;!?]', ' . ', line)
    line = re.sub('c.397delCinsGAA', ' c_397delCinsGAA ', line)
    line = re.sub(r'(\()(fc|si|ss|m)(\))([RD]NA)', r' \2\4 ', line)
    line = re.sub('Micro(mi)RNAs', r' miRNA ', line)
    line = re.sub(r'\)MAC', ') MAC', line)
    line = re.sub('dyness/cm5', ' dyness_cm5 ', line)
    line = re.sub(r'o\.6', ' 0.6 ', line)
    line = re.sub('C16H19N3O31.5H2O', 'C16H19N3O3__1_5H2O', line)
    line = re.sub('C20H22Br2N2O20.25CH4O', 'C20H22Br2N2O20_25CH4O', line)
    line = re.sub(r'{\[Zn2(C6H14N2O2)2(C10H8N2)3\]\(NO3\)40\.6H2O2C3H7NO}n',
                  '{[Zn2(C6H14N2O2)2(C10H8N2)3](NO3)40_6H2O2C3H7NO}n', line)
    line = re.sub(r' = 1-S0\(5\)EXP{0.9744 \(Risk Score - 2.3961\)}', '', line)
    line = re.sub(r'\(?GP\)?IIb/IIIa', 'GPIIb_IIa', line)
    line = re.sub(r'\(?CPP\)?-?PNA705', 'CPP_PNA705', line)
    line = re.sub(r'P\(a-et\)CO2', 'PaCO2 PETCO2', line)

    line = re.sub(r'([0-9])(yow/)', r'\1 years old with ', line)
    line = re.sub(r'(FSH)(-)([0-9])', r'\1 \3', line)
    line = re.sub(r'([<>=)])(\.[0-9]+)', r' \2 ', line)
    line = re.sub(r'(\.)(\[)', r'\1 \2', line)

    # sss
    line = re.sub('intusssuception', 'intussuception', line)
    line = re.sub('asssociated', 'associated', line)
    line = re.sub('classsified', 'classified', line)
    line = re.sub(' ssssss', ' ', line)

    line = re.sub(r'crosssectional|cross sectional', ' cross_sectional ', line)

    line = re.sub(r' (witness|routine|is|as|heterogeneous|autologous|endogenous|this|class)(s)+ ', r' \1 ', line)
    line = re.sub(r's{3,} ', ' ', line)

    line = re.sub(r'([ \t])(s)+([s|S](ome|ignificant|etting|tu|urfactant|yndrome|ample|almonella|pecialize|'
                  r'treptococcus|erum))', r'\1\3', line)

    line = re.sub(r'([ \t])(s){2,}(The |Despite |Assault |Eligible |From |[0-9]+ )', r'\1\3', line)
    line = re.sub(r'([ \t])(s){3,}([a-zA-z])', r'\1\3', line)

    # break words
    line = re.sub(r'([0-9a-zA-Z.,:)\]])(Final|Symptoms|Medication|Adverse|Clinic|Un|Prospective|Rare|Specialty|Cha|'
                  r'Congenital|Patient|Abdominal|Retrospective|Anatomical|Diagnostic|Objective|Mistake|Management|'
                  r'Acute|Schematic|Strongyloidiasis|Program|Catalogue|Licensing|No\. of |Distribution|Computer|'
                  r'Operating|RAM|Classification|External|Nature)', r'\1 \2', line)

    line = re.sub(r'(therapy|disease[s]?|diagnosis|Introduction|Interventions|course|Background|Objective[s]?|'
                  r'Discussion[s]?|Conclusion[s]?|Case|study|variation|accidents|treatment|ology|care|malpractice|'
                  r'reaction|infarction|Hypertension|presentation|atient[s]?|Medicine|Purpose|Methods|Results|\.)'
                  r'([A-Z])', r'\1 \2', line)

    line = re.sub(r'([a-z])(This|The|Although|Main)', r'\1. \2', line)

    # abbreviations with dot
    line = re.sub(r' i\.?v\.? ', ' intravenous ', line)
    line = re.sub(r'i\.e\.,?', ' i_e ', line)
    line = re.sub(r'e\.g\.,?', ' e_g ', line)
    line = re.sub(r's\.d\.', ' s_d', line)
    line = re.sub(r'r\.m\.s\.,?', ' r_m_s ', line)
    line = re.sub(r'b\.i\.d\.?|BID', ' b_i_d ', line)
    line = re.sub(r'q\.i\.d\.?|QID', ' q_i_d ', line)
    line = re.sub(r'q\.d\.?', ' q_d ', line)
    line = re.sub(r'm\.a\.s\.l\.', ' m_a_s_l ', line)
    line = re.sub(r'ClinicalTrials?\.(gov|org)', ' ClinicalTrials ', line)
    line = re.sub(r'([0-9] )([ap])(\.?m\.? )', r' \1\2_m ', line)  # a.m. p.m.
    line = re.sub(r'vs\.', 'versus', line)

    # remove random strings
    line = re.sub(r'(\\documentclass).+?(?=\\end{document})\\end{document}', ' ', line)  # remove TeX
    line = re.sub(r'(http|https)://[^\s\\[\]()]*', '', line)  # remove websites called after '.The'
    line = re.sub(r'www.[^\s\\[\]()]*', '', line)  # remove websites called after '.The'
    line = re.sub(r'(ISRCTN|NCT|ANZCTR|LOC) ?[0-9]+', ' ', line)
    line = re.sub(r'(doi|DOI): ?[0-9./a-zA-Z\-]+', ' ', line)
    line = re.sub(r'ISBN [0-9\- ]+', ' ', line)
    line = re.sub(r'ChiCTR-TRC[0-9]+', ' ', line)

    line = re.sub(r'PM\(10\)', ' PM10 ', line)
    line = re.sub(r'PM[(]?2.5[)]?', ' PM2_5 ', line)

    line = re.sub(r'[<>=,:|+&@~^#%*\"\'\\/`]', '', line)

    # remove ' (...) ' but cannot remove nested parentheses
    line = re.sub(r'\([ ]*\)', '', line)
    line = re.sub(r'([ \t])(\([ a-zA-Z0-9._\-[\]]*\))([ .\n\]])', r'\1\3', line)
    line = re.sub(r'([ \t])(\([ a-zA-Z0-9._\-[\]]*\))([ .\n\]])', r'\1\3', line)
    line = re.sub(r'([ \t])(\([ a-zA-Z0-9._\-[\]]*\))([ .\n\]])', r'\1\3', line)
    line = re.sub(r'([ \t])(\([ a-zA-Z0-9._\-[\]]*\))([ .\n\]])', r'\1\3', line)
    line = re.sub(r'([ \t])(\[[ a-zA-Z0-9._\-()]*\])([ .\n])', r'\1\3', line)
    line = re.sub(r'([ \t])(\[[ a-zA-Z0-9._\-()]*\])([ .\n])', r'\1\3', line)
    line = re.sub(r'([ \t])(\[[ a-zA-Z0-9._\-()]*\])([ .\n])', r'\1\3', line)
    line = re.sub(r'([ \t])(\[[ a-zA-Z0-9._\-()]*\])([ .\n])', r'\1\3', line)

    # numbers
    line = re.sub(r'([0-9])(year|month|h |mg |day|week|hour|cm |min |kg |mm |m |g |s |ft|lbs|L)', r'\1 \2', line)
    line = re.sub(r'([0-9])(-)(year|month|week|day|hour|min|minute|second|fold)([s]?[- ])', r'\1 \3 ', line)
    line = re.sub(r'([0-9])(year|y |yrs|yr|yo)', r'\1 year', line)

    line = re.sub(r'([0-9]+-){2,}[0-9]+', ' ', line)  # XXX-XXX-XXX or more
    line = re.sub(r'[0-9]+(\.[0-9]+)?-[0-9]+(\.[0-9]+)?', ' numbers_decimals ', line)  # XX-XX
    line = re.sub(r'[0-9]+(\.[0-9]+)?\([0-9]+\)', ' numbers_decimals ', line)  # XX(XX)
    line = re.sub(r'([ \t])([0-9]+[ .():])*([0-9]+)([ .\n])', r'\1numbers_decimals\4', line)
    line = re.sub(r' \.[0-9]+ ', ' numbers_decimals ', line)

    line = re.sub(r'\.', ' . ', line)
    line = re.sub(r'([ \t])([a-zA-Z]|[0-9]+)([ .\n])', r'\1\3', line)  # remove terms with single character and digits
    line = re.sub(r'\.( ?\.)+', '.', line)

    pcm_id = line.split('\t')[0]
    text = line.split('\t')[1]
    text = ' '.join(text.split())  # remove multiple spaces

    return pcm_id + '\t' + text + '\n'


def clean_docs(file_name):
    start = time.time()
    with open(file_name) as f:
        docs = [line for line in f]

    cleaned = Parallel(n_jobs=-1)(delayed(clean_line)(doc) for doc in docs)

    save_list('data/' + file_name.split('/')[1].split('.')[0] + '_cleaned.txt', cleaned)

    print("Finished cleaning data. Time: {}".format(time.time() - start))


if __name__ == '__main__':
    dataset_test()
