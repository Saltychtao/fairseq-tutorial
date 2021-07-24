databin=./data/tgt-data-bin
savedir=./savedir/finetuned-lstm
src_checkpoint_path=/path/to/src_checkpoint_path
fairseq-train \
    $databin \
    --arch lstm\
    --optimizer adam --adam-betas '(0.9, 0.98)' --finetune-from-model $src_checkpoint_path\
    --lr 5e-5 \
    --encoder-layers 1 --encoder-bidirectional --decoder-layers 1\
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 4000\
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 20}' \
    --eval-bleu-detok moses --eval-bleu-print-samples\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric\
    --no-save-optimizer-state  --no-epoch-checkpoints --patience 10 --save-dir $savedir -s de -t en\
