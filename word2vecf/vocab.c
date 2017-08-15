#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vocab.h"
#include "io.h"

#define MAX_STRING 100

//{{{ Hash
#include "stdint.h" /* Replace with <stdint.h> if appropriate */
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif

const int vocab_hash_size = 50000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

// hash from: http://www.azillionmonkeys.com/qed/hash.html
uint32_t FastHash(const char * data, int len) {
uint32_t hash = len, tmp;
int rem;

    if (len <= 0 || data == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
} //}}}


// Returns hash value of a word
inline int GetWordHash(struct vocabulary *v, char *word) {
  unsigned long long hash = 0;
  char *b = word;
  //for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  //hash = FastHash(word, strlen(word)) % vocab_hash_size;
  while (*b != 0) hash = hash * 257 + *(b++);
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(struct vocabulary *v, char *word) {
  unsigned int hash = GetWordHash(v, word);
  while (1) {
    if ((v->vocab_hash)[hash] == -1) return -1;
    if (!strcmp(word, v->vocab[v->vocab_hash[hash]].word)) return v->vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(struct vocabulary *v, char *word) {
  //static long collide = 0;
  //static long nocollide = 0;
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  v->vocab[v->vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(v->vocab[v->vocab_size].word, word);
  v->vocab[v->vocab_size].cn = 0;
  v->vocab_size++;
  // Reallocate memory if needed
  if (v->vocab_size + 2 >= v->vocab_max_size) {
    v->vocab_max_size += 1000;
    v->vocab = (struct vocab_word *)realloc(v->vocab, v->vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(v, word);
  //if (v->vocab_hash[hash] != -1) { collide += 1; } else { nocollide += 1; }
  //if ((collide + nocollide) % 100000 == 0) printf("%d %d %f collisions\n\n",collide, nocollide, (float)collide/(collide+nocollide));
  while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  v->vocab_hash[hash] = v->vocab_size - 1;
  return v->vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortAndReduceVocab(struct vocabulary *v, int min_count) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&(v->vocab[1]), v->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
  size = v->vocab_size;
  v->word_count = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (v->vocab[a].cn < min_count) {
      v->vocab_size--;
      free(v->vocab[v->vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(v, v->vocab[a].word);
      while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      v->vocab_hash[hash] = a;
      v->word_count += v->vocab[a].cn;
    }
  }
  v->vocab = (struct vocab_word *)realloc(v->vocab, (v->vocab_size + 1) * sizeof(struct vocab_word));
}

struct vocabulary *CreateVocabulary() {
   struct vocabulary *v = malloc(sizeof(struct vocabulary));
   long long a;
   v->vocab_max_size = 1000;
   v->vocab_size = 0;

   v->vocab = (struct vocab_word *)calloc(v->vocab_max_size, sizeof(struct vocab_word));

   v->vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
   for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
   return v;
}

void SaveVocab(struct vocabulary *v, char *save_vocab_file) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < v->vocab_size; i++) fprintf(fo, "%s %lld\n", v->vocab[i].word, v->vocab[i].cn);
  fclose(fo);
}
// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct vocabulary *v) {
   static int min_reduce = 1;
   printf("reducevocab\n");
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < v->vocab_size; a++) if (v->vocab[a].cn > min_reduce) {
    v->vocab[b].cn = v->vocab[a].cn;
    v->vocab[b].word = v->vocab[a].word;
    b++;
  } else free(v->vocab[a].word);
  v->vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
  for (a = 0; a < v->vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(v, v->vocab[a].word);
    while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    v->vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(struct vocabulary *v, FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin, MAX_STRING);
  if (feof(fin)) return -1;
  return SearchVocab(v, word);
}

struct vocabulary *ReadVocab(char *vocabfile) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(vocabfile, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  struct vocabulary *v = CreateVocabulary();
  while (1) {
    ReadWord(word, fin, MAX_STRING);
    if (feof(fin)) break;
    a = AddWordToVocab(v, word);
    fscanf(fin, "%lld%c", &v->vocab[a].cn, &c);
    i++;
  }
  SortAndReduceVocab(v, 0);
  printf("Vocab size: %d\n", v->vocab_size);
  printf("Word count: %lld\n", v->word_count);
  return v;
}

void EnsureVocabSize(struct vocabulary *vocab) {
    if (vocab->vocab_size > vocab_hash_size * 0.7) ReduceVocab(vocab);
}
