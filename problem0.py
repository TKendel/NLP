import re
import numpy as np
import matplotlib.pyplot as plt

from nltk import FreqDist
from nltk.corpus import brown
from string import punctuation
from collections import Counter, defaultdict


brown_text = brown.words()
news_text = brown.words(categories='news')
reviews_text = brown.words(categories='reviews')

# Sort word count per given text in descending order
def sortWordCount(words):
    no_capitals = set([word.lower() for word in words])
    translate_table = dict((ord(char), None) for char in punctuation)
    no_punct = [s.translate(translate_table) for s in no_capitals]
    wordcounter = defaultdict(int)
    for word in no_punct:
        if word in wordcounter:
            wordcounter[word] += 1
        else:
            wordcounter[word] = 1
    sorting = [(k, wordcounter[k])for k in sorted(wordcounter, key = wordcounter.get, reverse = True)]
    return sorting

# Sort token count per given text in descending order
def sortTokenCount(text):
    fdist = FreqDist(w.lower() for w in text)
    sorted_freq = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True))
    print(sorted_freq.items())

brown_sorted = sortWordCount(brown_text)
news_sorted = sortWordCount(news_text)
reviews_sorted = sortWordCount(reviews_text)

# Number of tokens
fdist = FreqDist(brown_text)
print(f"Number of tokens: {fdist.N()}")

# Number of types
print('Total Categories:', len(brown.categories()))


nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
filtered = [w for w in brown_text if nonPunct.match(w)]
counts = Counter(filtered)

# Word count
print(f"Word count: {len(counts)}")

# Average number of words per sentance
sentances = brown.sents()
word_count = 0
for sentance in sentances:
    word_count += len(sentance)
print(f"Average number of words per sentance: { word_count/len(sentances)}")

# Average length of words
word_length = 0
for word in brown_text:
    word_length += len(word)
print(f'Average length of words: {word_length/len(brown_text)}')

# POS
POS_sorted = [word[1] for word in brown.tagged_words()]
# the most frequent words
print(f"The top 10 most frequent POS tags are : {Counter(POS_sorted).most_common(10)}")

# Write to output file
output = open("output.txt", "a")
output.write(f"Number of tokens: {fdist.N()}\n")
output.write(f'Total Categories: {len(brown.categories())}\n')
output.write(f"Word count: {len(counts)}\n")
output.write(f"Average number of words per sentance:{ word_count/len(sentances)}\n")
output.write(f'Average length of words: {word_length/len(brown_text)}\n')
output.write(f"The top 10 most frequent POS tags are : {Counter(POS_sorted).most_common(10)}")
output.close()

list_of_texts = [brown_sorted, news_sorted, reviews_sorted]
for index, text in enumerate(list_of_texts):
    ranks = []
    freqs = []
    for rank, word in enumerate(text):
        ranks.append(rank+1)
        freqs.append(word[1])

    plt.plot(freqs, ranks)
    if index == 0:
        plt.savefig('brown.png')
        plt.clf()
        plt.loglog(freqs,ranks)
        plt.savefig('brown_log.png')
    elif index == 1:
        plt.savefig('news.png')
        plt.clf()
        plt.loglog(freqs,ranks)
        plt.savefig('news_log.png')
    else:
        plt.savefig('reviews.png')
        plt.clf()
        plt.loglog(freqs,ranks)
        plt.savefig('reviews_log.png')
    plt.clf()
