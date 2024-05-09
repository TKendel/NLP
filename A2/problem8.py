import re
import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
from math import log

nltk.download('brown')
from nltk.corpus import brown

# Frequency distribution
brown_text = brown.words()
nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
filtered = [w for w in brown_text if nonPunct.match(w)]
freq_dist = FreqDist(filtered)

# Keep words occurring at least 10 times
filtered_words = set(word for word in freq_dist if freq_dist[word] >= 10)

cfdist = nltk.ConditionalFreqDist()

# Update conditional frequencies
for w1, w2 in nltk.bigrams(filtered):
    w1_lower = w1.lower()
    w2_lower = w2.lower()
    if w1_lower in filtered_words and w2_lower in filtered_words:
        cfdist[w1_lower][w2_lower] += 1

# Total number of bigrams
total_bigrams = 0
for w1 in cfdist:
    total_bigrams += cfdist[w1].N()

# PMI
pmi_dict = {}
for w1 in cfdist:
    for w2 in cfdist[w1]:
        p_w1 = freq_dist[w1] / len(filtered)
        p_w2 = freq_dist[w2] / len(filtered)
        p_w1w2 = cfdist[w1][w2] / total_bigrams
        pmi = log(p_w1w2 / (p_w1 * p_w2), 2)
        pmi_dict[(w1, w2)] = pmi

# Sort PMIs
sorted_pmi = sorted(pmi_dict.items(), key=lambda item: item[1], reverse=True)

top_20_pmi = sorted_pmi[:20]
bottom_20_pmi = sorted_pmi[-20:]
print("Top 20:", top_20_pmi)
print("Bottom 20:", bottom_20_pmi)
