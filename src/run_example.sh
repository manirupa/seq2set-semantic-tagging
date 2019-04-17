#PBS -N EXP_NAME
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=28:gpus=1:default
qstat -f $PBS_JOBID


module load python cuda/9.0.176
source activate local
cd $PBS_O_WORKDIR


# download data.zip
# wget -O data.zip https://osu.box.com/s/eld6atk3y7m9923ael3ewkzmy22c3v08
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
  --model Transformer \
  --loss_fn lm \
  --keep_model_files True \
  --labels_path data/labels.pickle \
  --num_epochs 60 \
  --batch_size 32 \
  --dropout 0.2 \
  --alpha 1.0 \
  --lr 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --epsilon 1e-9 \
  --doc_vec_length 50 \
  --term_size 9956 \
  --mlp_layer_dims 50 1000 2500 5000 9956 \
  --hidden_size 200 \
  --num_layers 2 \
  --attention_size 200 \
  --num_heads 5 \
  --max_length 256 \
  --padding_length 256
