from joblib import Parallel, delayed
from nltk.corpus import stopwords
from textblob import TextBlob

from utils import save_list

STOP_WORDS = stopwords.words("english")


def embed_phrases(line, phrase_list):
    if not line or not phrase_list:
        return line

    token_list = line.split()

    token_count = len(token_list)
    phrase_count = len(phrase_list)

    cur_phrase = 0
    cur_token = 0

    embedded_tokens = []

    while cur_token < token_count:
        if cur_phrase == phrase_count:
            embedded_tokens.extend(token_list[cur_token:])
            break
        phrase = phrase_list[cur_phrase].split()
        cur_phrase_len = len(phrase)

        # matching all tokens of a phrase
        if token_list[cur_token: cur_token + cur_phrase_len] == phrase:
            embedded_tokens.append('_'.join(phrase))
            # advance token counter by length of phrase
            cur_token += cur_phrase_len
            cur_phrase += 1
        else:
            embedded_tokens.append(token_list[cur_token])
            cur_token += 1
    return ' '.join(embedded_tokens)


def main(input_dir):
    out_file = input_dir.replace('.txt', '_phrase_embedded.txt')

    def process_one_docs(line):
        temp = line.split("\t")
        pmc_id = temp[0]

        # clean data
        text = temp[1].strip('\n. ').lower()  # remove '\n', white spaces and the last '.'

        # remove stop words
        text = ' '.join(
            [word for word in text.split()
             if word not in STOP_WORDS])

        blob = TextBlob(text)
        phrases = [np for np in blob.noun_phrases if 1 <= np.count(' ') <= 2]
        new_line = embed_phrases(text, phrases) if len(phrases) > 0 else text

        # test
        # num_underscore = sum(1 for word in new_line.split() if '_' in word)
        # num_phrases = len(phrases)
        # assert num_underscore == num_phrases

        return pmc_id + '\t' + new_line + '\n'

    with open(input_dir, encoding='utf-8') as input_file:
        new_documents = Parallel(n_jobs=-1)(
            delayed(process_one_docs)(line) for line in input_file)

    save_list(out_file, new_documents)
