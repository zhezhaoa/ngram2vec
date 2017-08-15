#ifndef _vocab_h
#define _vocab_h


struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct vocabulary {
   struct vocab_word *vocab;
   int *vocab_hash;
   long long vocab_max_size; //1000
   long vocab_size;
   long long word_count;
};


int ReadWordIndex(struct vocabulary *v, FILE *fin);
inline int GetWordHash(struct vocabulary *v, char *word);
int SearchVocab(struct vocabulary *v, char *word);
int AddWordToVocab(struct vocabulary *v, char *word);
void SortAndReduceVocab(struct vocabulary *v, int min_count);
struct vocabulary *CreateVocabulary();
void SaveVocab(struct vocabulary *v, char *vocab_file);
struct vocabulary *ReadVocab(char *vocab_file);
void EnsureVocabSize(struct vocabulary *v);

#endif
