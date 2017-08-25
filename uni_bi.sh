#!/bin/sh

win=2
size=300
thr=100
sub=1e-5
iters=3
threads=10
negative=5
memsize=8.0
corpus=wiki2010.clean
output_path=outputs/uni_bi/win${win}

mkdir -p ${output_path}/sgns
mkdir -p ${output_path}/ppmi
mkdir -p ${output_path}/svd
python ngram2vec/corpus2vocab.py --ngram 2 --memory_size ${memsize} --min_count ${thr} ${corpus} ${output_path}/vocab
python ngram2vec/corpus2pairs.py --win ${win} --sub ${sub} --ngram_word 1 --ngram_context 2 --threads_num ${threads} --overlap ${corpus} ${output_path}/vocab ${output_path}/pairs
if [ -f "${output_path}/pairs" ]
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

#SGNS evaluation
cp ${output_path}/words.vocab ${output_path}/sgns/sgns.words.vocab
cp ${output_path}/contexts.vocab ${output_path}/sgns/sgns.contexts.vocab
python ngram2vec/text2numpy.py ${output_path}/sgns/sgns.words
python ngram2vec/text2numpy.py ${output_path}/sgns/sgns.contexts
for dataset in testsets/analogy/google.txt testsets/analogy/semantic.txt testsets/analogy/syntactic.txt testsets/analogy/msr.txt
do
	python ngram2vec/analogy_eval.py SGNS ${output_path}/sgns/sgns ${dataset}
done
for dataset in testsets/ws/ws353_similarity.txt testsets/ws/ws353_relatedness.txt testsets/ws/bruni_men.txt testsets/ws/radinsky_mturk.txt testsets/ws/luong_rare.txt testsets/ws/sim999.txt
do
	python ngram2vec/ws_eval.py SGNS ${output_path}/sgns/sgns ${dataset}
done

python ngram2vec/pairs2counts.py --memory_size ${memsize} ${output_path}/pairs ${output_path}/words.vocab ${output_path}/contexts.vocab ${output_path}/counts 
python ngram2vec/counts2ppmi.py ${output_path}/words.vocab ${output_path}/contexts.vocab ${output_path}/counts ${output_path}/ppmi/ppmi

#PPMI evaluation
for dataset in testsets/analogy/google.txt testsets/analogy/semantic.txt testsets/analogy/syntactic.txt testsets/analogy/msr.txt
do
	python ngram2vec/analogy_eval.py PPMI ${output_path}/ppmi/ppmi ${dataset}
done
for dataset in testsets/ws/ws353_similarity.txt testsets/ws/ws353_relatedness.txt testsets/ws/bruni_men.txt testsets/ws/radinsky_mturk.txt testsets/ws/luong_rare.txt testsets/ws/sim999.txt
do
	python ngram2vec/ws_eval.py PPMI ${output_path}/ppmi/ppmi ${dataset}
done

python ngram2vec/ppmi2svd.py ${output_path}/ppmi/ppmi ${output_path}/svd/svd 

#SVD evaluation
cp ${output_path}/words.vocab ${output_path}/svd/svd.words.vocab
cp ${output_path}/contexts.vocab ${output_path}/svd/svd.contexts.vocab
for dataset in testsets/analogy/google.txt testsets/analogy/semantic.txt testsets/analogy/syntactic.txt testsets/analogy/msr.txt
do
	python ngram2vec/analogy_eval.py SVD ${output_path}/svd/svd ${dataset}
done
for dataset in testsets/ws/ws353_similarity.txt testsets/ws/ws353_relatedness.txt testsets/ws/bruni_men.txt testsets/ws/radinsky_mturk.txt testsets/ws/luong_rare.txt testsets/ws/sim999.txt
do
	python ngram2vec/ws_eval.py SVD ${output_path}/svd/svd ${dataset}
done
