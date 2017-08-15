#!/bin/sh

win=2
size=300
thr=10
sub=1e-5
iters=3
threads=12
negative=5
corpus=wiki.en.txt
output_path=outputs/uni_uni

mkdir -p outputs/uni_uni/win${win}/sgns
python hyperwords/corpus2vocab.py --ngram 1 ${corpus} ${output_path}/win${win}/vocab
python hyperwords/corpus2pairs.py --win ${win} --sub ${sub} --ngram_word 1 --ngram_context 1 --thr ${thr} --threads ${threads} ${corpus} ${output_path}/win${win}/vocab ${output_path}/win${win}/pairs

if [ -f "${output_path}/win${win}/pairs" ]
then
	rm ${output_path}/win${win}/pairs
fi
for i in $(seq 0 $((${threads}-1)) )
do
	cat ${output_path}/win${win}/pairs_${i} >> ${output_path}/win${win}/pairs
	rm ${output_path}/win${win}/pairs_${i}
done

python hyperwords/pairs2vocab.py ${output_path}/win${win}/pairs ${output_path}/win${win}/sgns/sgns.words.vocab ${output_path}/win${win}/sgns/sgns.contexts.vocab
./word2vecf/word2vecf -train ${output_path}/win${win}/pairs -pow 0.75 -cvocab ${output_path}/win${win}/sgns/sgns.contexts.vocab -wvocab ${output_path}/win${win}/sgns/sgns.words.vocab -dumpcv ${output_path}/win${win}/sgns/sgns.contexts -output ${output_path}/win${win}/sgns/sgns.words -threads ${threads} -negative ${negative} -size ${size} -iters ${iters}


python hyperwords/text2numpy.py ${output_path}/win${win}/sgns/sgns.words
python hyperwords/text2numpy.py ${output_path}/win${win}/sgns/sgns.contexts

python hyperwords/analogy_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/analogy/google.txt
python hyperwords/analogy_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/analogy/semantic.txt
python hyperwords/analogy_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/analogy/syntactic.txt
python hyperwords/analogy_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/analogy/msr.txt

python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/ws353_similarity.txt
python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/ws353_relatedness.txt
python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/bruni_men.txt
python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/radinsky_mturk.txt
python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/luong_rare.txt
python hyperwords/ws_eval.py SGNS ${output_path}/win${win}/sgns/sgns testsets/ws/sim999.txt


