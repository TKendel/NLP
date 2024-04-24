import numpy as np

vocab_file = 'brown_vocab_100.txt'
corpus_file = 'brown_100.txt'


def load_vocabulary(filename):
    vocab = set()
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip()
            vocab.add(word)
    return vocab


def bigrams_trigrams_counts(corpus_file, vocab):
    bigram_counts = {}
    trigram_counts = {}

    with open(corpus_file, 'r') as file:
        for line in file:
            words = line.lower().strip().split()

            filtered_words = []

            for word in words:
                if word in vocab:
                    filtered_words.append(word)

            # Remove start and end tokens
            filtered_words = filtered_words[1:-1]

            for i in range(2, len(filtered_words)):
                trigram = (filtered_words[i-2],
                           filtered_words[i-1], filtered_words[i])
                bigram = (filtered_words[i-2], filtered_words[i-1])

                if trigram in trigram_counts:
                    trigram_counts[trigram] += 1
                else:
                    trigram_counts[trigram] = 1

                if bigram in bigram_counts:
                    bigram_counts[bigram] += 1
                else:
                    bigram_counts[bigram] = 1

    return bigram_counts, trigram_counts


def calculate_probability(w1, w2, w3, bigram_counts, trigram_counts, vocab_size, alpha=0.1, smoothed=True):
    bigram = (w1, w2)
    trigram = (w1, w2, w3)
    trigram_count = trigram_counts.get(trigram, 0)
    bigram_count = bigram_counts.get(bigram, 0)
    if smoothed:
        return (trigram_count + alpha) / (bigram_count + alpha * vocab_size)
    else:
        return trigram_count / bigram_count if bigram_count != 0 else 0


vocab = load_vocabulary(vocab_file)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print("---------------------")

bigram_counts, trigram_counts = bigrams_trigrams_counts(corpus_file, vocab)
print(f"Bigram count: {len(bigram_counts)}")
print(f"Trigram count: {len(trigram_counts)}")
print("---------------------")


probabilities = [
    ('in', 'the', 'past'),
    ('in', 'the', 'time'),
    ('the', 'jury', 'said'),
    ('the', 'jury', 'recommended'),
    ('jury', 'said', 'that'),
    ('agriculture', 'teacher', ',')
]

for w1, w2, w3 in probabilities:
    unsmoothed_prob = calculate_probability(
        w1, w2, w3, bigram_counts, trigram_counts, vocab_size, smoothed=False)

    print(f"Unsmoothed p({w3} | {w1}, {w2}): {unsmoothed_prob:.6f}")
    print("---------------------")

print("\n")

for w1, w2, w3 in probabilities:
    smoothed_prob = calculate_probability(
        w1, w2, w3, bigram_counts, trigram_counts, vocab_size, smoothed=True)

    print(f"Smoothed p({w3} | {w1}, {w2}): {smoothed_prob:.6f}")
    print("---------------------")
