CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy word2vecf count_and_filter

count_and_filter: count_and_filter.c vocab.c io.c
	$(CC) vocab.c count_and_filter.c io.c -o count_and_filter $(CFLAGS)
word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2vecf : word2vecf.c vocab.c io.c
	$(CC) word2vecf.c vocab.c io.c -o word2vecf $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy count_and_filter word2vecf
