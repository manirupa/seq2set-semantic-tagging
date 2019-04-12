./multift skipgram \
    -input "data/cleaned_1_line.txt" \
    -output "modelfiles/multi_seq2set_e10_d300_vs2e-4_lr1e-5_margin1" \
    -lr 1e-5 -dim 300 \
    -ws 10 -epoch 10 -minCount 1 -loss ns -bucket 2000000 \
    -minn 3 -maxn 6 -thread 62 -t 1e-5 -lrUpdateRate 100 -multi 1 -var_scale 2e-4 -margin 1