#!/bin/sh

win=2
size=300
thr=100
sub=1e-5
iters=3
threads=8
negative=5
memsize=8.0
corpus=wiki2010.clean
output_path=outputs/bi_bi/win${win}

mkdir -p ${output_path}/sgns
#python ngram2vec/corpus2vocab.py --ngram 2 --memory_size ${memsize} --min_count ${thr} ${corpus} ${output_path}/vocab
#python ngram2vec/corpus2pairs.py --win ${win} --sub ${sub} --ngram_word 2 --ngram_context 2 --threads_num ${threads} --overlap ${corpus} ${output_path}/vocab ${output_path}/pairs
if [ -f "${output_path}/win${win}/pairs" ]
then
	rm ${output_path}/pairs
fi
for i in $(seq 0 $((${threads}-1)) )
do
	cat ${output_path}/pairs_${i} >> ${output_path}/pairs
	rm ${output_path}/pairs_${i}
done
python ngram2vec/pairs2vocab.py ${output_path}/pairs ${output_path}/words.vocab ${output_path}/contexts.vocab

./word2vecf/word2vecf -train ${output_path}/pairs -pow 0.75 -cvocab ${output_path}/contexts.vocab -wvocab ${output_path}/words.vocab -dumpcv ${output_path}/sgns/sgns.contexts -output ${output_path}/sgns/sgns.words -threads ${threads} -negative ${negative} -size ${size} -iters ${iters}


python ngram2vec/text2numpy.py ${output_path}/sgns/sgns.words
python ngram2vec/text2numpy.py ${output_path}/sgns/sgns.contexts
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

