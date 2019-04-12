import pickle


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
