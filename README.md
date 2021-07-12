# Fairseq
Fairseq是Facebook出品的、专为序列生成服务的深度学习库。Fairseq中实现了许多先进的序列生成算法，并且代码整洁，封装程度高，拓展性强，故受到了很多研究人员的欢迎。 本篇指南旨在对Fairseq库做一个简单的介绍，以帮助同学们快速熟悉这个强大的代码库，来方便之后进行自然语言生成方向的研究、实践。

## 简单使用
在深入了解Fairseq的代码结构之前，让我们先通过一个简单的示例，来熟悉一下Fairseq的使用流程，对应的数据和代码可以见附带的压缩包。

假设我们现在有一个中英翻译的数据集（见 data/chinese-english），数据集中包含了100000句对应的中英句对以供训练，并分别包含1000句中英句对以供验证和测试。

### 数据预处理
使用Fairseq的第一步是将原始数据预处理成二进制文件存储下来，以方便后续处理的方便。 为此，我们首先需要将原始的句对组织成 xxx.src, xxx.tgt的形式，xxx.src中存储了平行句对的源端句子，xxx.tgt中存储了平行句对的目标端句子，两个文件的每一行是一一对应的(data目录下已经进行了相应的处理)。然后，就可以使用以下指令

``` shell
fairseq-preprocess --source-lang zh --target-lang en --trainpref ./data/chiense-english/train --validpref ./data/chinese-english/valid --testpref ./data/chinese-english/test --destdir ./data/data-bin --workers 20
```
来处理数据，并将结果存放在 `./data/data-bin`下。

### 训练模型
将数据处理完毕之后，我们就可以使用Fairseq所实现的序列到序列算法来训练一个中文到英文的机器翻译模型。训练一个基于LSTM的Seq2Seq模型的命令已经写在了`train_lstm.sh`中，可以通过运行`CUDA_VISIBLE_DEVICES=x bash train_lstm.sh`执行。 让我们看一下`train_lstm.sh`的内容：

``` shell
databin=./data/data-bin
savedir=./savedir/lstm
fairseq-train \
    $databin \
    --arch lstm\
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 1e-3 \
    --encoder-layers 1 --encoder-bidirectional --decoder-layers 1\
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 4000\
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 20}' \
    --eval-bleu-detok moses --eval-bleu-print-samples\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric\
    --no-save-optimizer-state  --no-epoch-checkpoints --patience 10 --save-dir $savedir -s src -t tgt\

```
其中， `--arch`后面跟的`lstm`代表我们想要使用Fairseq中实现的`lstm`这个架构来进行模型的搭建（如果有需要的话，后续也可以自定义架构。） `--optimizer adam`表示我们希望使用Adam优化器来优化我们的模型。 `--encoder-layers 1 --encoder-bidirectional --decoder-layers 1 --dropout 0.1` 为模型的一些超参数, `--criterion label_smoothed_cross_entropy --label-smoothing 0.1` 表示我们希望使用带有Label Smoothing机制的交叉熵损失来作为模型的优化目标。除此之外还有很多参数，同学们可以自己去阅读源码来了解它们的含义。

同理，训练一个基于Transformer的Seq2Seq模型的命令如下:

``` shell
databin=./data/data-bin
savedir=./savedir/transformer
fairseq-train \
    $databin \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 \
    --eval-bleu --eval-bleu-print-samples\
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 20}' \
    --eval-bleu-detok moses \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric\
    --no-save-optimizer-state --patience 10 --save-dir $savedir -s src -t tgt \
```
与上面类似，同学们可以通过执行`CUDA_VISIBLE_DEVICES=x bash train_transformer.sh`运行。

### 测试训练好的模型
在上一个小节的训练完成后，我们就可以尝试使用训练得到的模型来进行翻译（当然，由于本教程中用来训练模型的平行语料规模相对较小，因此所获得模型的翻译性能可能并不会特别好）。在Fairseq中，进行翻译有两种方式，一种是直接使用`fairseq-generate`命令来翻译之前使用`fairseq-preprocess`命令处理好的数据集，示例的命令如下，可通过运行`CUDA_VISIBLE_DEVICES=x bash generate.sh`来执行:

``` shell
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

```
上述命令表示我们希望使用`$checkpoint_dir`这个目录下训练得到的`checkpoint_best.pt`这个模型来对`$databin`中的`test`子集进行翻译，翻译时使用的`beam size`为5，`batch-size`为128。 翻译好的结果被存储在了`$checkpoint_dir/beam-5/generate-test.txt`这个文件中。

当然，很多时候我们在处理数据的时候并不知道希望翻译的句子，所以需要在训练完模型之后，给定一个句子时，对其进行实时地翻译，这可以通过以下命令实现，可通过运行`CUDA_VISIBLE_DEVICES=x bash interactive.sh`来执行:

``` shell
databin=$1
checkpoint_dir=$2
result_path=$3
fairseq-interactive $databin \
                 --path $checkpoint_dir/checkpoint_best.pt \
                 --batch-size 128 \
                 --beam $5  \
                 --gen-subset $4 \
                 --remove-bpe --tokenizer moses
```
执行这个命令会进入一个交互式的界面，此时当输入一句待翻译的句子时，程序就能够调用训练好的模型来对其进行翻译，并输出结果。

下图是一个示例:
![./figs/interactive.png](./figs/interactive.png)
