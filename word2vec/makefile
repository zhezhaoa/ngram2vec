CC = gcc
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)

clean:
	rm -rf word2vec
