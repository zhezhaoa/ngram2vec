CC = gcc
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: glove

glove : glove.c
	$(CC) glove.c -o glove $(CFLAGS)

clean:
	rm -rf glove

