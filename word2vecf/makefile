CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vecf

word2vecf : word2vecf.c vocab.c io.c
	$(CC) word2vecf.c vocab.c io.c -o word2vecf $(CFLAGS)

clean:
	rm -rf word2vecf
