//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
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
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define MAX_STRING_LENGTH 1000

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int write_header=1; //0=no, 1=yes; writes vocab_size/vector_size as first line for use with some libraries, such as gensim.
int verbose = 2; // 0, 1, or 2
int use_unk_vec = 0; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
//!@#$%^&*
int vector_size = 300; // Word vector size
int use_binary = 0; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 1; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
int checkpoint_every = 0; // checkpoint the model for every checkpoint_every iterations. Do nothing if checkpoint_every <= 0
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *gradsq, *cost;
long long input_vocab_size, output_vocab_size, counts_num; //!@#$%^&*
char *counts_file, *input_vector_file, *output_vector_file, *input_vocab_file, *output_vocab_file; //!@#$%^&*
long long file_size = 0; //!@#$%^&*
/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
    long long a, b;
    vector_size++; // Temporarily increment to allocate space for bias

    /* Allocate space for word vectors and context word vectors, and correspodning gradsq */
    a = posix_memalign((void **)&W, 128, (input_vocab_size + output_vocab_size) * vector_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, (input_vocab_size + output_vocab_size) * vector_size * sizeof(real)); // Might perform better than malloc
	if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }
    for (b = 0; b < vector_size; b++) for (a = 0; a < input_vocab_size + output_vocab_size; a++) W[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    for (b = 0; b < vector_size; b++) for (a = 0; a < input_vocab_size + output_vocab_size; a++) gradsq[a * vector_size + b] = 1.0; // So initial value of eta is equal to initial learning rate
    vector_size--;
}

inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a, l1, l2;
    long long id = *(long long*)vid;
    CREC cr;
    real diff, fdiff, temp1, temp2;
    FILE *fin;
    fin = fopen(counts_file, "rb");
    cost[id] = 0;
    char c1, c2;
    
    long long start_offset = file_size / (long long)num_threads * (long long)id;
    long long end_offset = file_size / (long long)num_threads * (long long)(id+1);

    real* W_updates1 = (real*)malloc(vector_size * sizeof(real));
    real* W_updates2 = (real*)malloc(vector_size * sizeof(real));
    fseek(fin, start_offset, SEEK_SET);
    while (fgetc(fin) != '\n') { };

    while (1) {
        if (feof(fin) || ftell(fin) > end_offset) break;

        fscanf(fin, "%d%c%d%c%lf", &cr.word1, &c1, &cr.word2, &c2, &cr.val);

        if (feof(fin)) break;
        
        /* Get location of words in W & gradsq */
        l1 = cr.word1 * (vector_size + 1); // cr word indices start at 1
        l2 = (cr.word2 + input_vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words
        
        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (a = 0; a < vector_size; a++) diff += W[a + l1] * W[a + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff

        // Check for NaN and inf() in the diffs.
        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)) {
            fprintf(stderr,"Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }

        cost[id] += 0.5 * fdiff * diff; // weighted squared error
        
        /* Adaptive gradient updates */
        fdiff *= eta; // for ease in calculating gradient
        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        for (a = 0; a < vector_size; a++) {
            // learning rate times gradient for word vectors
            temp1 = fdiff * W[a + l2];
            temp2 = fdiff * W[a + l1];
            // adaptive updates
            W_updates1[a] = temp1 / sqrt(gradsq[a + l1]);
            W_updates2[a] = temp2 / sqrt(gradsq[a + l2]);
            W_updates1_sum += W_updates1[a];
            W_updates2_sum += W_updates2[a];
            gradsq[a + l1] += temp1 * temp1;
            gradsq[a + l2] += temp2 * temp2;
        }
        if (!isnan(W_updates1_sum) && !isinf(W_updates1_sum) && !isnan(W_updates2_sum) && !isinf(W_updates2_sum)) {
            for (a = 0; a < vector_size; a++) {
                W[a + l1] -= W_updates1[a];
                W[a + l2] -= W_updates2[a];
            }
        }
        // updates for bias terms
        W[vector_size + l1] -= check_nan(fdiff / sqrt(gradsq[vector_size + l1]));
        W[vector_size + l2] -= check_nan(fdiff / sqrt(gradsq[vector_size + l2]));
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
    }
    free(W_updates1);
    free(W_updates2);
    
    fclose(fin);
    pthread_exit(NULL);
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

/* Save params to file */
int save_params() {
    long long a, b;
    char format[20];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    FILE *fo_input, *fo_output, *fi_input_vocab, *fi_output_vocab;

    fo_input = fopen(input_vector_file, "wb");
    fo_output = fopen(output_vector_file, "wb");   
    fi_input_vocab = fopen(input_vocab_file, "r");
    fi_output_vocab = fopen(output_vocab_file, "r");

    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    fprintf(fo_input, "%lld %d\n", input_vocab_size, vector_size);

    for (a = 0; a < input_vocab_size; a++) { //!@#$%^&*
        if (fscanf(fi_input_vocab,format,word) == 0) return 1;
        fprintf(fo_input, "%s",word);
        for (b = 0; b < vector_size; b++) fprintf(fo_input," %lf", W[a * (vector_size + 1) + b]);
        fprintf(fo_input,"\n");
        if (fscanf(fi_input_vocab,format,word) == 0) return 1; // Eat irrelevant frequency entry
    }

    fprintf(fo_output, "%lld %d\n", output_vocab_size, vector_size);
    for (a = 0; a < output_vocab_size; a++) { //!@#$%^&*
        if (fscanf(fi_output_vocab,format,word) == 0) return 1;
        fprintf(fo_output, "%s",word);
        for (b = 0; b < vector_size; b++) fprintf(fo_output," %lf", W[(input_vocab_size + a) * (vector_size + 1) + b]);
        fprintf(fo_output,"\n");
        if (fscanf(fi_output_vocab,format,word) == 0) return 1; // Eat irrelevant frequency entry
    }

    fclose(fo_input);
    fclose(fo_output);
    fclose(fi_input_vocab);
    fclose(fi_output_vocab);
    return 0;
}

/* Train model */
int train_glove() {
    long long a;
    int b;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    file_size = GetFileSize(counts_file);
    fprintf(stderr,"Read %lld counts.\n", counts_num);
    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"words size: %lld\n", input_vocab_size);
    if (verbose > 0) fprintf(stderr,"contexts size: %lld\n", output_vocab_size);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    // Lock-free asynchronous SGD
    for (b = 0; b < num_iter; b++) {
        total_cost = 0;
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)&thread_ids[a]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        free(thread_ids);

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, cost: %lf\n", time_buffer,  b+1, total_cost/counts_num);

    }
    free(pt);
    return save_params();

}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    //!@#$%^&*
    counts_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    output_vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_vector_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    output_vector_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    int result = 0;
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 300\n");//!@#$%^&*
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        result = 0;
    } else {
	if ((i = find_arg((char *)"-write-header", argc, argv)) > 0) write_header = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"--verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"--size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"--iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"--threads_num", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        cost = malloc(sizeof(real) * num_threads);
        if ((i = find_arg((char *)"--alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = find_arg((char *)"--x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
        if ((i = find_arg((char *)"--eta", argc, argv)) > 0) eta = atof(argv[i + 1]);

        if ((i = find_arg((char *)"--counts_file", argc, argv)) > 0) strcpy(counts_file, argv[i + 1]);
        if ((i = find_arg((char *)"--input_vocab_file", argc, argv)) > 0) strcpy(input_vocab_file, argv[i + 1]);
        if ((i = find_arg((char *)"--output_vocab_file", argc, argv)) > 0) strcpy(output_vocab_file, argv[i + 1]);
        if ((i = find_arg((char *)"--input_vector_file", argc, argv)) > 0) strcpy(input_vector_file, argv[i + 1]);
        if ((i = find_arg((char *)"--output_vector_file", argc, argv)) > 0) strcpy(output_vector_file, argv[i + 1]);

        input_vocab_size = 0;
        fid = fopen(input_vocab_file, "r");
        if (fid == NULL) {fprintf(stderr, "Unable to open words file %s.\n",input_vocab_file); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') input_vocab_size++; // Count number of entries in vocab_file
        fclose(fid);

        output_vocab_size = 0;
        fid = fopen(output_vocab_file, "r");
        if (fid == NULL) {fprintf(stderr, "Unable to open output_vocab file %s.\n",output_vocab_file); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') output_vocab_size++; // Count number of entries in vocab_file
        fclose(fid);

        counts_num = 0;
        fid = fopen(counts_file, "r");
        if (fid == NULL) {fprintf(stderr, "Unable to open contexts file %s.\n",counts_file); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') counts_num++; // Count number of entries in vocab_file
        fclose(fid);
        
        result = train_glove();
        free(cost);
    }
    free(counts_file);
    free(input_vocab_file);
    free(output_vocab_file);
    free(input_vector_file);
    free(output_vector_file);
    return result;
}
