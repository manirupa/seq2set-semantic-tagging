"""
This beautiful automation script helps generate batch job scripts
for each model, given the set of hyperparameters to vary
"""

import string
import time
import datetime
import operator
import sys, re, os
from math import *

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H-%M-%S')

word_vec_files = ['pft_sg1_s50_w4_m1_i10.npy'] #, \
                  #'word2vec_sg1_s50_w4_m1_n5_i15.npy']
batch_sizes = [16, 32, 64]
dropouts = [0.2, 0.3, 0.4]
label_neighbors = [5, 10, 15, 20]
num_layers = [4] #2, 3]
losses = ['sigmoid', 'softmax_uniform']
alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
num_heads = [5, 10, 20]
models = ['BiLSTMATT'] 

EP=4
NH=20
NL=3
emb = word_vec_files[0] 
ALPHA = 1.0

def write_to_file(str_to_print,filename):
    f = open(filename, 'a')
    f.write(str_to_print)
    f.close()

def make_batch_script_file(jobname, model, loss, ALPHA, ln, emb, EP, bs, dp, NL, NH):
     script_text = "#PBS -N %s\n" % jobname
     script_text += "#PBS -j oe\n"
     script_text += "#PBS -l walltime=48:00:00\n"
     script_text += "#PBS -l nodes=1:ppn=24:gpus=1:default\n"
     script_text += "#PBS -A PAS1339\n"
     script_text += "qstat -f $PBS_JOBID\n"
     script_text += "\n\nmodule load python cuda/9.0.176\n"
     script_text += "source activate py36\n"
     script_text += "cd $PBS_O_WORKDIR\n\n"
     script_text += "# run\n"
     script_text += "python encode_and_inference.py \\\n"
     script_text += "  --model %s \\\n" % model
     script_text += "  --loss_fn %s \\\n" % loss
     script_text += "  --alpha %s \\\n" % ALPHA
     script_text += "  --keep_model_files false \\\n"
     script_text += "  --labels_path data/%s_label_neighbors_mc1 \\\n" % ln
     script_text += "  --word_vecs_path data/%s \\\n" % emb
     script_text += "  --docs_path data/docs_word_indices_mc1.pickle \\\n"
     script_text += "  --num_epochs %s \\\n" % EP
     script_text += "  --batch_size %s \\\n" % bs
     script_text += "  --dropout %s \\\n" % dp
     script_text += "  --lr 0.0001 \\\n"
     script_text += "  --beta1 0.9 \\\n"
     script_text += "  --beta2 0.999 \\\n"
     script_text += "  --epsilon 1e-9 \\\n"
     script_text += "  --doc_vec_length 50 \\\n"
     script_text += "  --term_size 9956 \\\n"
     script_text += "  --mlp_layer_dims 50 1000 2500 5000 9956 \\\n"
     script_text += "  --hidden_size 200 \\\n"
     script_text += "  --num_layers %s \\\n" % NL
     script_text += "  --attention_size 200 \\\n"
     script_text += "  --num_heads %s \\\n" % NH
     script_text += "  --max_length 256 \\\n"
     script_text += "  --padding_length 256\n"
     return script_text

batch_scripts_folder_name = './batch_scripts'

for model in models:
    #make a model dir for batch scripts for that model
    folder_name = '%s/%sjobs' % (batch_scripts_folder_name, model)
    try:
        os.system('mkdir -p %s' % folder_name)
    except:
        pass
    qsub_filename = '%s/%sjobs.sh' % (folder_name, model)
    
    #make each batch script to put in folder using hyperparams
    for bs in batch_sizes:
      for dp in dropouts:
         for ln in label_neighbors:
           for nl in num_layers:
              for loss in losses:
                 ts = time.time()
                 timestmp = str(int(round(ts)))
                 #if (bs == 16):
                 #    EP = 8
                 #if (bs == 32):
                 #    EP = 5
                 if (model == 'Transformer'):
                     jobname = '%s_testparam_%s_nh%s_bs%s_%s_kln%s_dp%s_ep%s' % \
                        (model, loss, NH, bs, emb, ln, dp, EP)
                 else:
                     jobname = '%s_testparam_%s_bs%s_%s_kln%s_dp%s_ep%s' % \
                        (model, loss, bs, emb, ln, dp, EP)
             
                 batch_filename = '%s/%s.%s.sh' % (folder_name, jobname, timestmp) 
             
                 script_text = make_batch_script_file(jobname, model, loss, ALPHA, ln, emb, EP, bs, dp, NL, NH)
             
                 write_to_file(script_text, batch_filename)
                 #write to final qsub file
                 write_to_file('qsub %s\n' % os.path.basename(batch_filename), qsub_filename)

    
    #for lm loss
    loss = 'lm'
    dropouts = [0.2, 0.3]
                         
    for bs in batch_sizes:
      for dp in dropouts:
         for ln in label_neighbors:
           for nl in num_layers:
            #add for loop for alpha
            for alpha in alphas:
                 ts = time.time()
                 timestmp = str(int(round(ts)))
                 #if (bs == 16):
                 #    EP = 8
                 #if (bs == 32):
                 #    EP = 5
                 if (model == 'Transformer'):
                     jobname = '%s_testparam_%s_%s_nh%s_bs%s_%s_kln%s_dp%s_ep%s' % \
                        (model, loss, alpha, NH, bs, emb, ln, dp, EP)
                 else:
                     jobname = '%s_testparam_%s_%s_bs%s_%s_kln%s_dp%s_ep%s' % \
                            (model, loss, alpha, bs, emb, ln, dp, EP)
                 
                 batch_filename = '%s/%s.%s.sh' % (folder_name, jobname, timestmp) 
             
                 script_text = make_batch_script_file(jobname, model, loss, alpha, ln, emb, EP, bs, dp, NL, NH)
             
                 write_to_file(script_text, batch_filename)
                 #write to final qsub file
                 write_to_file('qsub %s\n' % os.path.basename(batch_filename), qsub_filename)
