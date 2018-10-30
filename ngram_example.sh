#!/bin/sh

memory_size=4
cpus_num=4
corpus=wikipedia
output_path=outputs/${corpus}/ngram_ngram

mkdir -p ${output_path}/sgns
mkdir -p ${output_path}/ppmi
mkdir -p ${output_path}/svd
mkdir -p ${output_path}/glove

python ngram2vec/corpus2vocab.py --corpus_file ${corpus} --vocab_file ${output_path}/vocab --memory_size ${memory_size} --feature ngram --order 2
python ngram2vec/corpus2pairs.py --corpus_file ${corpus} --pairs_file ${output_path}/pairs --vocab_file ${output_path}/vocab --processes_num ${cpus_num} --cooccur ngram_ngram --input_order 1 --output_order 2

# Concatenate pair files. 
if [ -f "${output_path}/pairs" ]; then
	rm ${output_path}/pairs
fi
for i in $(seq 0 $((${cpus_num}-1)))
do
	cat ${output_path}/pairs_${i} >> ${output_path}/pairs
	rm ${output_path}/pairs_${i}
done

# Generate input vocabulary and output vocabulary, which are used as vocabulary files for all models
python ngram2vec/pairs2vocab.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output

# SGNS, learn representation upon pairs.
# We add a python interface upon C code.
python ngram2vec/pairs2sgns.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/sgns/sgns.input --output_vector_file ${output_path}/sgns/sgns.output --threads_num ${cpus_num} --size 300

# SGNS evaluation.
python ngram2vec/similarity_eval.py --input_vector_file ${output_path}/sgns/sgns.input  --test_file testsets/similarity/ws353_similarity.txt --normalize
python ngram2vec/analogy_eval.py --input_vector_file ${output_path}/sgns/sgns.input --test_file testsets/analogy/semantic.txt --normalize

# Generate co-occurrence matrix from pairs.
python ngram2vec/pairs2counts.py --pairs_file ${output_path}/pairs --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --counts_file ${output_path}/counts --output_id --memory_size ${memory_size}

# PPMI, learn representation upon counts (co-occurrence matrix).
python ngram2vec/counts2ppmi.py --counts_file ${output_path}/counts --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --ppmi_file ${output_path}/ppmi/ppmi

# PPMI evaluation.
python ngram2vec/similarity_eval.py --input_vector_file ${output_path}/ppmi/ppmi --test_file testsets/similarity/ws353_similarity.txt --normalize --sparse
python ngram2vec/analogy_eval.py --input_vector_file ${output_path}/ppmi/ppmi --test_file testsets/analogy/semantic.txt --normalize --sparse

# SVD, factorize PPMI matrix.
python ngram2vec/ppmi2svd.py --ppmi_file ${output_path}/ppmi/ppmi --svd_file ${output_path}/svd/svd --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output 

# SVD evaluation.
python ngram2vec/similarity_eval.py --input_vector_file ${output_path}/svd/svd.input  --test_file testsets/similarity/ws353_similarity.txt --normalize
python ngram2vec/analogy_eval.py --input_vector_file ${output_path}/svd/svd.input --test_file testsets/analogy/semantic.txt --normalize

# Shuffle counts.
python ngram2vec/shuffle.py --input_file ${output_path}/counts --output_file ${output_path}/counts.shuf --memory_size ${memory_size}

# GloVe, learn representation upon counts (co-occurrence matrix).
python ngram2vec/counts2glove.py --counts_file ${output_path}/counts.shuf --input_vocab_file ${output_path}/vocab.input --output_vocab_file ${output_path}/vocab.output --input_vector_file ${output_path}/glove/glove.input --output_vector_file ${output_path}/glove/glove.output --threads_num ${cpus_num}

# GloVe evaluation.
python ngram2vec/similarity_eval.py --input_vector_file ${output_path}/glove/glove.input  --test_file testsets/similarity/ws353_similarity.txt --normalize
python ngram2vec/analogy_eval.py --input_vector_file ${output_path}/glove/glove.input --test_file testsets/analogy/semantic.txt --normalize
