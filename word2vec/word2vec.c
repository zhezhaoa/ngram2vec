// Modifed by ZheZhao, Renmin University of China 
// fix some bugs such as training info printing 
// delete irrelevant code
//
/////////////////////////////////////////////////////////////////
// TODO: add total word count to vocabulary, instead of "train_words"
//
// Modifed by Yoav Goldberg, Jan-Feb 2014
// Removed:
//    hierarchical-softmax training
//    cbow
// Added:
//   - support for different vocabularies for words and contexts
//   - different input syntax
//
/////////////////////////////////////////////////////////////////
//
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

const long long vocab_hash_size = 70000000;
typedef float real;

struct vocab_word {
  long long cn;
  char *word;
};

struct vocabulary {
   struct vocab_word *vocab;
   long long *vocab_hash;
   long long vocab_max_size; //1000
   long long vocab_size;
   long long pairs_num;
};

char pairs_file[MAX_STRING];
char input_vocab_file[MAX_STRING], output_vocab_file[MAX_STRING];
char input_vector_file[MAX_STRING], output_vector_file[MAX_STRING];
int debug_mode = 2, num_threads = 1;
long long vec_size = 100;
long long pairs_num = 0, pairs_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1neg, *expTable;
clock_t start;
int numiters = 1;

struct vocabulary *input_vocab;
struct vocabulary *output_vocab;

int negative = 15;
const long long table_size = 1e8;
long long *samplingtable;

void InitSamplingTable(struct vocabulary *v) {
  long long a, i;
  long long normalizer = 0;
  real d1, power = 0.75;
  samplingtable = (long long *)malloc(table_size * sizeof(long long));
  for (a = 0; a < v->vocab_size; a++) normalizer += pow(v->vocab[a].cn, power);
  i = 0;
  d1 = pow(v->vocab[i].cn, power) / (real)normalizer;
  for (a = 0; a < table_size; a++) {
    samplingtable[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(v->vocab[i].cn, power) / (real)normalizer;
    }
    if (i >= v->vocab_size) i = v->vocab_size - 1;
  }
}

void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) break;
      else continue; 
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;
  }
  word[a] = 0;
}

// Returns hash value of a word
unsigned long long GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}


long long SearchVocab(struct vocabulary *v, char *word) {
  unsigned long long hash = GetWordHash(word);
  while (1) {
    if ((v->vocab_hash)[hash] == -1) return -1;
    if (!strcmp(word, v->vocab[v->vocab_hash[hash]].word)) return v->vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Adds a word to the vocabulary
long long AddWordToVocab(struct vocabulary *v, char *word) {
  unsigned long long hash;
  int length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  v->vocab[v->vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(v->vocab[v->vocab_size].word, word);
  v->vocab[v->vocab_size].cn = 0;
  v->vocab_size++;
  if (v->vocab_size + 2 >= v->vocab_max_size) {
    v->vocab_max_size += 1000;
    v->vocab = (struct vocab_word *)realloc(v->vocab, v->vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (v->vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  v->vocab_hash[hash] = v->vocab_size - 1;
  return v->vocab_size - 1;
}

struct vocabulary *CreateVocabulary() {
   struct vocabulary *v = malloc(sizeof(struct vocabulary));
   long long a;
   v->vocab_max_size = 1000;
   v->vocab_size = 0;
   v->vocab = (struct vocab_word *)calloc(v->vocab_max_size, sizeof(struct vocab_word));
   v->vocab_hash = (long long *)calloc(vocab_hash_size, sizeof(long long));
   for (a = 0; a < vocab_hash_size; a++) v->vocab_hash[a] = -1;
   return v;
}

void SaveVocab(struct vocabulary *v, char *save_vocab_file) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < v->vocab_size; i++) fprintf(fo, "%s %lld\n", v->vocab[i].word, v->vocab[i].cn);
  fclose(fo);
}

// Reads a word and returns its index in the vocabulary
long long ReadWordIndex(struct vocabulary *v, FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
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
  v->pairs_num = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(v, word);
    fscanf(fin, "%lld%c", &v->vocab[a].cn, &c);
    v->pairs_num += v->vocab[a].cn;
    i++;
  }
  printf("Vocab size: %lld\n", v->vocab_size);
  printf("Number of pairs: %lld\n", v->pairs_num);
  return v;
}

long long GetFileSize(char *fname) {
  long long fsize;
  FILE *fin = fopen(fname, "rb");
  if (fin == NULL) {
    printf("ERROR: file not found! %s\n", fname);
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  fsize = ftell(fin);
  fclose(fin);
  return fsize;
}

void InitNet(struct vocabulary *input_vocab, struct vocabulary *output_vocab) {
   long long a, b;
   a = posix_memalign((void **)&syn0, 128, (long long)input_vocab->vocab_size * vec_size * sizeof(real));
   if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
   for (b = 0; b < vec_size; b++) 
      for (a = 0; a < input_vocab->vocab_size; a++)
         syn0[a * vec_size + b] = (rand() / (real)RAND_MAX - 0.5) / vec_size;

   a = posix_memalign((void **)&syn1neg, 128, (long long)output_vocab->vocab_size * vec_size * sizeof(real));
   if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
   for (b = 0; b < vec_size; b++)
      for (a = 0; a < output_vocab->vocab_size; a++)
        syn1neg[a * vec_size + b] = 0;
}

void *TrainModelThread(void *id) {
  long long input = -1, output = -1;
  long long d;
  long long pairs_count = 0, last_pairs_count = 0;
  long long l1, l2, c, target, label;
  unsigned long long next_random = (unsigned long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(vec_size, sizeof(real));
  real *neu1e = (real *)calloc(vec_size, sizeof(real));
  FILE *fi = fopen(pairs_file, "rb");
  long long start_offset = file_size / (long long)num_threads * (long long)id;
  long long end_offset = file_size / (long long)num_threads * (long long)(id+1);
  int iter;
  for (iter=0; iter < numiters; ++iter) {
     fseek(fi, start_offset, SEEK_SET);
     while (fgetc(fi) != '\n') { };
     long long pairs_num = input_vocab->pairs_num;
     while (1) {
        if (pairs_count - last_pairs_count > 10000) {
           pairs_count_actual += pairs_count - last_pairs_count;
           last_pairs_count = pairs_count;
           if ((debug_mode > 1)) {
              now=clock();
              printf("%cAlpha: %f  Progress: %.2f%%  Pairs/thread/sec: %.2fk  ", 13, alpha,
                    pairs_count_actual / (real)(numiters*pairs_num + 1) * 100,
                    pairs_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
              fflush(stdout);
           }
           alpha = starting_alpha * (1 - pairs_count_actual / (real)(numiters*pairs_num + 1));
           if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        if (feof(fi) || ftell(fi) > end_offset) break;
        for (c = 0; c < vec_size; c++) neu1[c] = 0;
        for (c = 0; c < vec_size; c++) neu1e[c] = 0;
        input = ReadWordIndex(input_vocab, fi);
        output = ReadWordIndex(output_vocab, fi);
        pairs_count++;
        if (input < 0 || output < 0) continue;
        // Negative sampling.
        l1 = input * vec_size;
        for (d = 0; d < negative + 1; d++) {
           if (d == 0) {
              target = output;
              label = 1;
           } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = samplingtable[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (output_vocab->vocab_size - 1) + 1;
              if (target == output) continue;
              label = 0;
           }
           l2 = target * vec_size;
           f = 0;
           for (c = 0; c < vec_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
           if (f > MAX_EXP) g = (label - 1) * alpha;
           else if (f < -MAX_EXP) g = (label - 0) * alpha;
           else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
           for (c = 0; c < vec_size; c++) neu1e[c] += g * syn1neg[c + l2];
           for (c = 0; c < vec_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        for (c = 0; c < vec_size; c++) syn0[c + l1] += neu1e[c];
     }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo1;
  FILE *fo2;
  file_size = GetFileSize(pairs_file);
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", pairs_file);
  starting_alpha = alpha;
  input_vocab = ReadVocab(input_vocab_file);
  output_vocab = ReadVocab(output_vocab_file);
  InitNet(input_vocab, output_vocab);
  InitSamplingTable(output_vocab);
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo1 = fopen(input_vector_file, "wb");
  fprintf(fo1, "%lld %lld\n", input_vocab->vocab_size, vec_size);
  for (a = 0; a < input_vocab->vocab_size; a++) {
    fprintf(fo1, "%s ", input_vocab->vocab[a].word);
    for (b = 0; b < vec_size; b++) fprintf(fo1, "%lf ", syn0[a * vec_size + b]);
    fprintf(fo1, "\n");
  }
  fclose(fo1);

  fo2 = fopen(output_vector_file, "wb");
  fprintf(fo2, "%lld %lld\n", output_vocab->vocab_size, vec_size);
  for (a = 0; a < output_vocab->vocab_size; a++) {
      fprintf(fo2, "%s ", output_vocab->vocab[a].word);
      for (b = 0; b < vec_size; b++) fprintf(fo2, "%lf ", syn1neg[a * vec_size + b]);
      fprintf(fo2, "\n");
  }
  fclose(fo2);
  
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit\n\n");
    printf("Options:\n");
    printf("Hyper-parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse pairs data from <file> to train the model\n");
    printf("\t-input_vocab <file>\n");
    printf("\t\tinput vocabulary file\n");
    printf("\t-output_vocab <file>\n");
    printf("\t\toutput vocabulary file\n");
    printf("\t-input_vector <file>\n");
    printf("\t\tUse <file> to save the resulting input vectors\n");
    printf("\t-output_vector <file>\n");
    printf("\t\tUse <file> to save the resulting output vectors\n");

    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 15, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-iters <int>\n");
    printf("\t\tPerform i iterations over the data; default is 1\n");
    return 0;
  }
  pairs_file[0] = 0;
  input_vocab_file[0] = 0;
  output_vocab_file[0] = 0;
  input_vector_file[0] = 0;
  output_vector_file[0] = 0;
  if ((i = ArgPos((char *)"--pairs_file", argc, argv)) > 0) strcpy(pairs_file, argv[i + 1]);
  if ((i = ArgPos((char *)"--input_vocab_file", argc, argv)) > 0) strcpy(input_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"--output_vocab_file", argc, argv)) > 0) strcpy(output_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"--input_vector_file", argc, argv)) > 0) strcpy(input_vector_file, argv[i + 1]);
  if ((i = ArgPos((char *)"--output_vector_file", argc, argv)) > 0) strcpy(output_vector_file, argv[i + 1]);
  if ((i = ArgPos((char *)"--size", argc, argv)) > 0) vec_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"--debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"--alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"--negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"--threads_num", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"--iter", argc, argv)) > 0) numiters = atoi(argv[i+1]);

  if (pairs_file[0] == 0) { printf("must supply -train.\n\n"); return 0; }
  if (input_vocab_file[0] == 0) { printf("must supply -input_vocab.\n\n"); return 0; }
  if (output_vocab_file[0] == 0) { printf("must supply -output_vocab.\n\n"); return 0; }
  if (input_vector_file[0] == 0) { printf("must supply -input_vector.\n\n"); return 0; }
  if (output_vector_file[0] == 0) { printf("must supply -output_vector.\n\n"); return 0; }
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
