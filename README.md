# ngram2vec
The toolkit implements the ngram2vec model proposed in emnlp 2017

In this work, we introduce ngrams into recent word representation methods inspired by traditional language modeling problem. Significant improvements are witnessed in some settings when ngrams are considered. For example, PPMI achieves 85+ accuracy on Google analogy questions (semantic group).This toolkit provides the workflow of the proposed ngram models. It enables users to build vocabulary and co-occurrence matrix at a certain memory size. Also, we do optimization on many stages to speed up the process and reduce disk space required.


**corpus2vocab** builds ngram vocabulary from corpus<br>
**corpus2pairs** extracts ngram pairs from corpus (multi-threading implementation), used by SGNS model.
**pairs2vocab** generates center word vocabulary and context vocabulary, which are used by all models. (note that the two vocabularies are different. In ·uni_bi· case Center word voca)
