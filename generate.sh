databin=$1
checkpoint_dir=$2
fairseq-generate $databin \
                 --path $checkpoint_dir/checkpoint_best.pt \
                 --batch-size 128 \
                 --beam 5  \
                 --results-path $checkpoint_dir/beam-5 \
                 --scoring sacrebleu \
                 --gen-subset test \
                 --tokenizer moses

tail -1 $checkpoint_dir/beam-5/generate-test.txt
