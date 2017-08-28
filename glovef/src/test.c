#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double real;
typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int main(int argc, char **argv) {
    //int i;
    //long long vocab_size;
    //FILE *f = fopen("contexts.vocab", "r");
    //while ((i = getc(f)) != EOF) if (i == '\n') vocab_size++; 
    //printf("vocab size: %lld\n", vocab_size);

    /*
    FILE *fi = fopen("counts.shuf", "r");
    int i = 0;
    long long tmp;
    i = fscanf(fi, "%lld", &tmp);
    printf("%d\n", i);
    printf("%lld\n", tmp);
    i = fscanf(fi, "%lld", &tmp);
    printf("%d\n", i);
    printf("%lld\n", tmp);
    i = fscanf(fi, "%lld", &tmp);
    printf("%d\n", i);
    printf("%lld\n", tmp);
    i = fscanf(fi, "%lld", &tmp);
    printf("%d\n", i);
    printf("%lld\n", tmp);
    i = fscanf(fi, "%lld", &tmp);
    printf("%d\n", i);
    printf("%lld\n", tmp);
    fscanf(fi, "%lld", &tmp);
    printf("%lld\n", tmp);
    fscanf(fi, "%lld", &tmp);
    printf("%lld\n", tmp);
    fscanf(fi, "%lld", &tmp);
    printf("%lld\n", tmp);
    fscanf(fi, "%lld", &tmp);
    printf("%lld\n", tmp);
    fclose(fi);
    */
    
    /*
    CREC cr;
    FILE *fin = fopen("counts.shuf.bin", "rb");
    long long cnt = 0;
    while(1){
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        //printf("%d %d %lf\n", cr.word1, cr.word2, cr.val);
        cnt += 1;
    }
    printf("%lld\n", cnt);
    fclose(fin);
    */

    /*
    FILE *fin = fopen("counts.shuf.bin", "rb");
    fseeko(fin, 0, SEEK_END);
    long long file_size = ftello(fin);
    long long num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    printf("file size: %lld\n", file_size);
    printf("number of counts: %lld\n", num_lines);
    */

}

