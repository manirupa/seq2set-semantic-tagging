# seq2set-semantic-tagging

## Packgages
- tensorflow 1.10.0

- tensor2tensor 1.10.0

- textblob 0.15.1

- joblib 0.12.5

- gensim 3.6.0

- numpy 1.15.2

- kaggle

- pytorch

## Pre-precessing

cleaned.txt: all lower case words, stop words removed

### Download data

```bash
cd data/  # assuming data/ is empty
Download all of the files from this URL -- https://osu.box.com/s/eld6atk3y7m9923ael3ewkzmy22c3v08
```
<!---kaggle datasets download -d --unzip joshdamen/seq2set
--->

### ELMo

```bash
python elmo.py \
    --input data/cleaned_no_ids.txt
    --output data/elmo_layers.hdf5
    --options data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    --weights data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    --cuda_device -1
    --batch_size 64
    --only_get_npy  # this is used when we already have the hdf5 file
```

### Word2vec and FastText

Get word embeddings and labels
```bash
python pre_process.py -i cleaned.txt -m fast_text
# Options for -m: fast_text, word2vec.
# Options for -i: cleaned.txt (unigram), cleaned_phrase_embedded.txt (phrase)

python make_labels_neighbors.py -l data/labels_mc2 -i data/index2word_mc2 -t data/terms_mc2 -n 3
```

### Probabilistic FastText

```bash
# compile
cd multisense_prob_fasttext
make

mkdir modelfiles
cd ..

export output=multisense_prob_fasttext/modelfiles/multi_seq2set_e10_d300_vs2e-4_lr1e-5_margin1
./multisense_prob_fasttext/multift skipgram \
    -input "data/cleaned_1_line.txt" \
    -output $output \
    -lr 1e-5 -dim 300 \
    -ws 10 -epoch 10 -minCount 1 -loss ns -bucket 2000000 \
    -minn 3 -maxn 6 -thread 62 -t 1e-5 \
    -lrUpdateRate 100 -multi 1 -var_scale 2e-4 -margin 1

python2 pft.py -i $output
```

### Get TFIDF only
Skip getting word embeddings:

1. In `model_params.py`, set:

```python
settings = {'sg': 1, 'size': 50, 'window': 4, 'min_count': 1, 'negative': 5, 'iter': 1}
```
adjust min_count accordingly.

2. Run:

```bash
python pre_process.py -i cleaned.txt -m word2vec
```

### Make labels neighbors

```bash
python make_labels_neighbors.py -l $labels_path \
	-i $index2word_path \
	-t $terms_path \
```

## Encode and inference

### Encode and inference

Note: Document Encoder Model options
```
--model: options: DAN, LSTM, BiLSTM, BiLSTMATT, Transformer, doc2vec
```

```bash
python encode_and_inference.py \
  --dropout=0.3 \
  --num_epochs=100 \
  --batch_size=128 \
  --model=BiLSTMATT \
  --loss_fn=sigmoid \
  --word_vecs_path=data/word2vec_sg1_s300_w5_m2_n5_i15.npy \
  --docs_path=data/docs_word_indices_mc2 \
  --labels_path=data/labels_mc2 \
  --doc_tfidf_reps_path=data/doc_tfidf_reps_mc2 \
  --index2word_path=data/index2word_mc2 \
  --terms_path=data/terms_mc2
```

or

```bash
python encode_and_inference.py \
  --model=doc2vec \
  --word_vecs_path=data/word2vec_sg1_s300_w5_m2_n5_i15.npy \
  --docs_path=data/docs_word_indices_mc2 \
  --labels_path=data/labels_mc2 \
  --doc_tfidf_reps_path=data/doc_tfidf_reps_mc2 \
  --index2word_path=data/index2word_mc2 \
  --terms_path=data/terms_mc2
```

### Direct inference

TODO
