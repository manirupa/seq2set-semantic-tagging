#PBS -N ondemand/sys/myjobs/basic_python_serial
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=28:gpus=1:default
#PBS -A PAS1339
qstat -f $PBS_JOBID


module load python cuda/9.0.176
source activate local
cd $PBS_O_WORKDIR


# download data.zip
# wget -O data.zip https://www.dropbox.com/s/3wv4x2ndql0n07s/data.zip?dl=1
# unzip data.zip
# rm data.zip


# test
# python encode_and_inference.py --model=LSTM --loss_fn=sigmoid --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=LSTM --loss_fn=softmax_uniform --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=LSTM --loss_fn=softmax_skewed_labels --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=LSTM --loss_fn=sigmoid_with_constraint --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=DAN --loss_fn=sigmoid --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=BiLSTM --loss_fn=sigmoid --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=BiLSTMATT --loss_fn=sigmoid --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=Transformer --loss_fn=sigmoid --num_epochs=2 --test_mode=1  # pass
# python encode_and_inference.py --model=doc2vec --test_mode=1  # pass


# run
python encode_and_inference.py \
  --dropout=0.3 \
  --num_epochs=100 \
  --batch_size=128 \
  --model=LSTM \
  --loss_fn=sigmoid \
  --word_vecs_path=data/word2vec_sg1_s300_w5_m2_n5_i15.npy \
  --docs_path=data/docs_word_indices_mc2 \
  --labels_path=data/labels_mc2 \
  --doc_tfidf_reps_path=data/doc_tfidf_reps_mc2 \
  --index2word_path=data/index2word_mc2 \
  --terms_path=data/terms_mc2
