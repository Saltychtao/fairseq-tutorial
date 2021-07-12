databin=$1
checkpoint_dir=$2
fairseq-interactive $databin \
                 --path $checkpoint_dir/checkpoint_best.pt \
                 --batch-size 1 \
                 --beam 5 \
                 --tokenizer moses --input -
