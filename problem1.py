#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
from collections import defaultdict


# TODO: read brown_vocab_100.txt into word_index_dict
def createBrownVocabDict(text):
    word_index_dict = {}

    for idx, word in enumerate(text):
        new_word = {word.rstrip('\n'): idx}
        word_index_dict.update(new_word)
    return word_index_dict

if __name__ == "__main__":
    text = open("brown_vocab_100.txt")
    word_index_dict = createBrownVocabDict(text)

    # TODO: write word_index_dict to word_to_index_100.txt
    wf = open('word_to_index_100.txt','w')
    wf.write(str(word_index_dict))
    wf.close()

    print(word_index_dict['all'])
    print(word_index_dict['resolution'])
    print(len(word_index_dict))
