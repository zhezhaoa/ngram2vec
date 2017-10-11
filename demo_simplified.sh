#!/bin/sh

win=2
size=300
thr=100
sub=1e-5
iters=3
threads=8
negative=5
corpus=wiki2010.clean
output_path=outputs/uni_uni/win${win}

mkdir -p ${output_path}/sgns
mkdir -p ${output_path}/ppmi
mkdir -p ${output_path}/svd
mkdir -p ${output_path}/glove
python ngram2vec/simplified/corpus2vocab.py --ngram 1 --min_count ${thr} ${corpus} ${output_path}/vocab
python ngram2vec/simplified/corpus2pairs.py --win ${win} --sub ${sub} --ngram_word 1 --ngram_context 1 ${corpus} ${output_path}/vocab ${output_path}/pairs
python ngram2vec/pairs2vocab.py ${output_path}/pairs ${output_path}/words.vocab ${output_path}/contexts.vocab

# word2vecf implemented by python
python word2vecf/word2vecf.py ${output_path}/pairs ${output_path}/words.vocab ${output_path}/contexts.vocab ${output_path}/sgns/sgns.words --negative ${negative} --size ${size} --iters ${iters} --processes_num ${threads}
python ngram2vec/text2numpy.py ${output_path}/sgns/sgns.words
analogy_path=testsets/analogy
for dataset in ${analogy_path}/google.txt ${analogy_path}/semantic.txt ${analogy_path}/syntactic.txt ${analogy_path}/msr.txt
do
	python ngram2vec/analogy_eval.py SGNS ${output_path}/sgns/sgns ${dataset}
done
ws_path=testsets/ws
for dataset in ${ws_path}/ws353_similarity.txt ${ws_path}/ws353_relatedness.txt ${ws_path}/bruni_men.txt ${ws_path}/radinsky_mturk.txt ${ws_path}/luong_rare.txt ${ws_path}/sim999.txt
do
	python ngram2vec/ws_eval.py SGNS ${output_path}/sgns/sgns ${dataset}
done
