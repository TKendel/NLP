#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np

from generate import GENERATE
from problem1 import createBrownVocabDict

vocab = open("brown_vocab_100.txt")

# load the indices dictionary
brown_vocab_dict = createBrownVocabDict(vocab)

f = open("brown_100.txt")

counts = np.zeros(len(brown_vocab_dict))

#  iterate through file and update counts
for sentence in f:
    # print(sentance)
    split_sentance = sentence.lower().split(" ")
    for token in split_sentance:
        vocab_id = brown_vocab_dict.get(token)
        if vocab_id == None:
            continue
        else:
            counts[vocab_id] = counts[vocab_id] + 1

f.close()

print(counts)
# normalize and writeout counts.
probs = counts / np.sum(counts)
wf = open('unigram_probs.txt', 'w')
wf.write(str(probs))
wf.close()

# Q6 : Calculating sentence proabilities

with open('unigram_eval.txt', 'w') as wf, open('toy_corpus.txt', 'r') as ty_file:
    for line in ty_file:
        split_line = line.lower().split()
        sentprob = 1
        sent_len = len(split_line)
        for token in split_line:
            vocab_id = brown_vocab_dict.get(token)
            if vocab_id is None:
                continue
            sentprob *= probs[vocab_id]
        perplexity = 1 / (pow(sentprob, 1.0 / sent_len))
        wf.write(str(perplexity) + '\n')
        print("Perplexity: ", perplexity)
        wf.write(str(sentprob) + '\n')
        print("Joint Prob: ", sentprob)


# Q7, sentence generation
filename = "unigram_generation.txt"
with open(filename, 'w') as f:
    for i in range(10):
        gen = GENERATE(brown_vocab_dict, probs, "unigram", 25, '<s>')
        print(gen)
        f.write(gen + "\n")
