from random import Random
import random
from corpus2vocab import getNgram


def ngram_ngram(line, args, vocab, pairs_file, sub, subsampler):
    win = int(args['--win'])
    ngram_word = int(args['--ngram_word'])
    ngram_context = int(args['--ngram_context'])
    overlap = args['--overlap']
    rnd = Random(17)
    tokens = line.strip().split()
    for i in range(len(tokens)): #loop for each position in a line
        for gram_word in range(1, ngram_word+1): #loop for grams of different orders in (center) word 
            word = getNgram(tokens, i, gram_word)
            word = check_word(word, vocab, sub, subsampler, rnd)
            if word is None:
                continue
            for gram_context in range(1, ngram_context+1): #loop for grams of different orders in context
                start = i - win + gram_word - 1
                end = i + win - gram_context + 1
                for j in range(start, end + 1):
                    if overlap:
                        if i == j and gram_word == gram_context:
                            continue
                    else:
                        if len(set(range(i, i + gram_word)) & set(range(j, j + gram_context))) > 0:
                            continue
                    context = getNgram(tokens, j, gram_context)
                    context = check_word(context, vocab, sub, subsampler, rnd)
                    if context is None:
                        continue
                    pairs_file.write(word + ' ' + context + "\n") #write pairs to the file


def word_word(line, args, vocab, pairs_file, sub, subsampler): #identical to the word2vec toolkit; dynamic and dirty window!
    win = int(args['--win'])
    win = random.randint(1, win) #dynamic window
    rnd = Random(17)
    tokens = [t if t in vocab else None for t in line.strip().split()]
    if sub:
        tokens = [t if t not in subsampler or rnd.random() > subsampler[t] else None for t in tokens]
    tokens = [t for t in tokens if t is not None] #dirty window
    for i in range(len(tokens)): #loop for each position in a line
        word = getNgram(tokens, i, 1)
        if word is None:
            continue
        start = i - win
        end = i + win
        for j in range(start, end + 1):
            if i == j:
                continue
            context = getNgram(tokens, j, 1)
            if context is None:
                continue
            pairs_file.write(word + ' ' + context + "\n")


def word_wordLR(line, args, vocab, pairs_file, sub, subsampler):
    win = int(args['--win'])
    rnd = Random(17)
    tokens = line.strip().split()
    for i in range(len(tokens)): #loop for each position in a line
        word = getNgram(tokens, i, 1)
        word = check_word(word, vocab, sub, subsampler, rnd)
        if word is None:
            continue
        start = i - win
        end = i + win
        for j in range(start, end + 1):
            if i == j:
                continue
            context = getNgram(tokens, j, 1)
            context = check_word(context, vocab, sub, subsampler, rnd)
            if context is None:
                continue
            if j < i:
                pairs_file.write(word + ' ' + context + '#L' + "\n")
            else:
                pairs_file.write(word + ' ' + context + '#R' + "\n")


def word_wordPos(line, args, vocab, pairs_file, sub, subsampler):
    win = int(args['--win'])
    rnd = Random(17)
    tokens = line.strip().split()
    for i in range(len(tokens)): #loop for each position in a line
        word = getNgram(tokens, i, 1)
        word = check_word(word, vocab, sub, subsampler, rnd)
        if word is None:
            continue
        start = i - win
        end = i + win
        for j in range(start, end + 1):
            if i == j:
                continue
            context = getNgram(tokens, j, 1)
            context = check_word(context, vocab, sub, subsampler, rnd)
            if context is None:
                continue
            if j < i:
                pairs_file.write(word + ' ' + context + '#L' + str(i-j) + "\n")
            else:
                pairs_file.write(word + ' ' + context + '#R' + str(j-i) + "\n")


def word_text(line, args, vocab, pairs_file, sub, subsampler, text_id):
    rnd = Random(17)
    tokens = line.strip().split()
    if len(tokens) < 200:
        return
    for i in range(len(tokens)): #loop for each position in a line
        word = getNgram(tokens, i, 1)
        word = check_word(word, vocab, sub, subsampler, rnd)
        if word is None:
            continue
        pairs_file.write('#' + str(text_id) + ' ' + word + "\n")


def check_word(t, vocab, sub, subsampler, rnd): #discard tokens
    if t is None:
        return None
    if sub:
        t = t if t not in subsampler or rnd.random() > subsampler[t] else None
        if t is None:
            return None
    t = t if t in vocab else None
    return t
