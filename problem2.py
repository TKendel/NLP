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

#load the indices dictionary
brown_vocab_dict = createBrownVocabDict(vocab)

f = open("brown_100.txt")

counts = np.zeros(len(brown_vocab_dict))

#TODO: iterate through file and update counts
for sentance in f:
    # print(sentance)
    split_sentance = sentance.lower().split(" ")
    for token in split_sentance:
        vocab_id = brown_vocab_dict.get(token)
        if vocab_id == None:
            continue
        else:
            counts[vocab_id] = counts[vocab_id] + 1

f.close()

print(counts)
#TODO: normalize and writeout counts. 
probs = counts / np.sum(counts)
wf = open('unigram_probs.txt','w')
wf.write(str(probs))
wf.close()


