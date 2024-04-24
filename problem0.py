import re
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

# sortWordCount(news_text), sortWordCount(brown_text), sortWordCount(reviews_text)

brown_sorted = sortWordCount(brown_text)


# Number of tokens
fdist = FreqDist(brown_text)
print(fdist.N())

# Number of types
print('Total Categories:', len(brown.categories()))


nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
filtered = [w for w in brown_text if nonPunct.match(w)]
counts = Counter(filtered)

# Word count
print(len(counts))

# Average number of words per sentance
sentances = brown.sents()
word_count = 0
for sentance in sentances:
    word_count += len(sentance)
print(word_count/len(sentances))

# Average length of words
word_length = 0
for word in brown_text:
    word_length += len(word)
print(word_length/len(brown_text))

# POS
POS_sorted = [word[1] for word in brown.tagged_words()]
# the most frequent words
print(Counter(POS_sorted).most_common(10))

x, y = zip(*brown_sorted) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.savefig('test.png')