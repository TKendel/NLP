#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
from problem1 import createBrownVocabDict

# import part 1 code to build dictionary
with open("brown_vocab_100.txt") as file:
    vocab = file.read().splitlines()

brown_vocab_dict = createBrownVocabDict(vocab)

# initialize numpy 0s array
vocab_size = len(brown_vocab_dict)
bigram_counts = np.zeros((vocab_size, vocab_size))


# iterate through file and update counts
def update_bigram_counts(filename):
    previous_word = '<s>'
    with open(filename, 'r') as file:
        for line in file:
            words = line.lower().strip().split()
            for word in words:
                if word in brown_vocab_dict and previous_word in brown_vocab_dict:
                    bigram_counts[brown_vocab_dict[previous_word]][brown_vocab_dict[word]] += 1
                previous_word = word


update_bigram_counts("brown_100.txt")

# smoothing
bigram_counts += 0.1

# normalize counts
bigram_probs = normalize(bigram_counts, norm='l1', axis=1)

# writeout bigram probabilities
with open("smooth_probs.txt", "w") as f:
    bigrams_to_write = [('all', 'the'), ('the', 'jury'), ('the', 'campaign'), ('anonymous', 'calls')]
    for previous, word in bigrams_to_write:
        if previous in brown_vocab_dict and word in brown_vocab_dict:
            prob = bigram_probs[brown_vocab_dict[previous]][brown_vocab_dict[word]]
            f.write(f"p({word} | {previous}) = {prob}\n")


# Q6 : Calculating sentence probabilities

with open('smoothed_eval.txt', 'w') as wf, open('toy_corpus.txt', 'r') as ty_file:
    for line in ty_file:
        split_line = line.lower().split()[:-1]
        print("split_line", split_line)
        sentprob = 1
        sent_len = len(split_line) 
        print("Processing sentence:", split_line)
        for i in range(len(split_line) - 1):
            previous = split_line[i]
            word = split_line[i + 1]
            print("Processing bigram:", previous, word)
            if previous in brown_vocab_dict and word in brown_vocab_dict:
                vocab_id_previous = brown_vocab_dict[previous]
                vocab_id_word = brown_vocab_dict[word]
                print("Vocab IDs:", vocab_id_previous, vocab_id_word)
                sentprob *= bigram_probs[vocab_id_previous][vocab_id_word]
                print("Sentprob:", sentprob)
        perplexity = 1 / (pow(sentprob, 1.0/sent_len))
        print("Perplexity:", perplexity)
        wf.write(str(perplexity) + '\n')


# Q7, sentence generation
filename = "smoothed_generation.txt"
with open(filename, 'w') as f:
    for i in range(10):
        gen = GENERATE(brown_vocab_dict, bigram_probs, "bigram", 25, '<s>')
        print(gen)
        f.write(gen + "\n")
