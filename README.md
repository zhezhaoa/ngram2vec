# Ngram2vec
Ngram2vec toolkit is originally used for reproducing results of the paper
<a href="http://www.aclweb.org/anthology/D17-1023"><em>Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics</em></a>
, aiming at learning high quality word embedding and ngram embedding.

Thansks to its well-designed architecture (we will talk about it later), ngram2vec toolkit provides a general and powerful framework, which is able to include researches of a large amount of papers and many popular toolkits such as word2vec. Ngram2vec toolkit allows researchers to learn representations upon co-occurrence statistics easily. Ngram2vec can generate embeddings of different granularities (beyond word embedding). For example, ngram2vec toolkit could be used for learning text embedding. Text embeddings trained by ngram2vec are very competitive. They outperform many deep and complex neural networks and achieve state-of-the-art results on a range of datasets. More details will be released later. 

Ngram2vec has been successfully applied on many projects. For example, <a href="https://github.com/Embedding/Chinese-Word-Vectors"><em>Chinese-Word-Vectors</em></a> provides over 100  Chinese word embeddings with different properties. All embeddings are trained by ngram2vec toolkit.

The original version (v0.0.0) of ngram2vec can be downloaded on github release. Python2 is recommended. One can download ngram2vec v0.0.0 for reproducing results.

## Features
Ngram2vec is featured by decoupled architecture. The process from raw corpus to final embeddings is decoupled into multiple modules. This brings many advantages compared with other toolkits.
* Well-organized: The ngram2vec toolkit is easy to read and understand.
* Extensibility: One can add co-occurrence statistics and embedding models with little effort.
* Intermediate results reuse: Intermediate results are written to disk and reused later, which largely boosts the efficiency in both speed and space.
* Comprehensive: Ngram2vec includes a large amount of works related with word embedding
* Embeddings of different linguistic units: Ngram2vec can learn embeddings of different linguistic units. For example, ngram2vec is able to produce high-quality text embeddings which achieve SOTA reults on a range of datasets. 


## Requirements
* Python (both Python2 and 3 are supported)
* numpy
* scipy
* sparsesvd

## Example use cases

Firstly, run the following codes to make some files executable.<br>
`chmod +x *.sh`<br>
`chmod +x scripts/clean_corpus.sh`<br>
`python scripts/compile_c.py`<br>

Also, a corpus should be prepared. We recommend to fetch it at<br> 
http://nlp.stanford.edu/data/WestburyLab.wikicorp.201004.txt.bz2 , a wiki corpus without XML tags. `scripts/clean_corpus.sh` is used for cleaning English corpus.<br> For example `scripts/clean_corpus.sh WestburyLab.wikicorp.201004.txt > wiki2010.clean`<br>
A pre-processed (including segmentation) chinese wiki corpus is available at https://pan.baidu.com/s/1kURV0rl , which can be directly used as input of this toolkit.

run `./word_example.sh` to see baselines<br>
run `./ngram_example.sh` to introduce ngram into recent word representation methods inspired by traditional language modeling problem.br>

## Workflow

<img src="https://github.com/zhezhaoa/ngram2vec/blob/master/workflow.jpg" width = "600" align=center />

## Testsets

Besides English word analogy and similarity datasets, we provide several **Chinese** analogy datasets, which contain comprehensive analogy questions. Some of them are constructed by directly translating English analogy datasets. Some are unique to Chinese. I hope they could become useful resources for evaluating Chinese word embedding.


## References

    @inproceedings{DBLP:conf/emnlp/ZhaoLLLD17,
         author = {Zhe Zhao and Tao Liu and Shen Li and Bofang Li and Xiaoyong Du},
         title = {Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics},   
         booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2017, Copenhagen, Denmark, September 9-11, 2017},      
         year = {2017}
     }


## Acknowledgments

This toolkit is inspired by Omer Levy's work http://bitbucket.org/omerlevy/hyperwords<br>
We reuse part of his code in this toolkit. We also thank him for his kind suggestions.<br>
I also got the help from Bofang Li, [Prof. Ju Fan](http://info.ruc.edu.cn/academic_professor.php?teacher_id=162), and Jianwei Cui in Xiaomi.<br>
My tutors are [Tao Liu](http://info.ruc.edu.cn/academic_professor.php?teacher_id=46) and [Xiaoyong Du](http://info.ruc.edu.cn/academic_professor.php?teacher_id=57)

## Contact us

We are looking forward to receiving your questions and advice to this toolkit. We will reply you as soon as possible. We will further perfect this toolkit.<br>  
[Zhe Zhao](https://zhezhaoa.github.io/), helloworld@ruc.edu.cn, from [DBIIR lab](http://iir.ruc.edu.cn/index.jsp)<br>
Shen Li, shen@mail.bnu.edu.cn<br>
Renfen Hu, irishere@mail.bnu.edu.cn
