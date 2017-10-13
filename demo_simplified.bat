set win=2
set size=100
set thr=100
set sub=1e-5
set iters=2
set negative=5
set memsize=4
set corpus=news.2012.en.shuffled
set output_path=outputs\\uni_uni\\win%win%

mkdir %output_path%\\sgns
mkdir %output_path%\\ppmi
mkdir %output_path%\\svd
mkdir %output_path%\\glove

python ngram2vec\\simplified\\corpus2vocab.py --ngram 1 --min_count %thr% %corpus% %output_path%\\vocab
python ngram2vec\\simplified\\corpus2pairs.py --win %win% --sub %sub% --ngram_word 1 --ngram_context 1 %corpus% %output_path%\\vocab %output_path%\\pairs
python ngram2vec\\pairs2vocab.py %output_path%\\pairs %output_path%\\words.vocab %output_path%\\contexts.vocab

python ngram2vec\\simplified\\word2vecf.py %output_path%\\pairs %output_path%\\words.vocab %output_path%\\contexts.vocab %output_path%\\sgns\\sgns.words --negative %negative% --size %size% --iters %iters%

copy %output_path%\\words.vocab %output_path%\\sgns\\sgns.words.vocab
python ngram2vec\\text2numpy.py %output_path%\\sgns\\sgns.words

set analogy_path=testsets/analogy
python ngram2vec\\analogy_eval.py SGNS %output_path%\\sgns\\sgns %analogy_path%\\google.txt
python ngram2vec\\analogy_eval.py SGNS %output_path%\\sgns\\sgns %analogy_path%\\semantic.txt
python ngram2vec\\analogy_eval.py SGNS %output_path%\\sgns\\sgns %analogy_path%\\syntactic.txt
python ngram2vec\\analogy_eval.py SGNS %output_path%\\sgns\\sgns %analogy_path%\\msr.txt

set ws_path=testsets/ws
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\ws353_similarity.txt
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\ws353_relatedness.txt
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\bruni_men.txt
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\radinsky_mturk.txt
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\luong_rare.txt
python ngram2vec\\ws_eval.py SGNS %output_path%\\sgns\\sgns %ws_path%\\sim999.txt

rem python ngram2vec/pairs2counts.py --memory_size %memsize% %output_path%\\pairs %output_path%\\words.vocab %output_path%\\contexts.vocab %output_path%\\counts

pause


