src-data-bin=/path/to/src-data-bin
tgt-data-path=/path/to/tgt-data
srclang=de
tgtlang=en
fairseq-preprocess --source-lang de --target-lang en --trainpref $tgt-data-path/train --validpref .$tgt-data-path/valid --testpref $tgt-data-path/test --destdir ./data/tgt-data-bin --workers 20 --srcdict $src-data-bin/dict.de.txt --tgtdict $tgt-data-bin/dict.en.txt
