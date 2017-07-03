import numpy as np
import collections
import random
from scipy.special import expit


class TGSG():
    def __init__(self, sentences, doc_sents,dim=100, neg_samples=5, alpha=0.025, window=10, seed=1, hashfxn=hash):
        self.alpha = alpha
        self.window = window
        self.seed = seed
        self.dim = dim
        self.hashfxn = hashfxn
        self.random = np.random.RandomState(seed)
        self.data = sentences
        self.doc_data = doc_sents
        self.vocab = dict()
        self.vocab2 = dict()
        self.index2word = dict()
        self.index2word2 = dict()
        self.word2index = dict()
        self.word2index2 = dict()
        self.build_vocab(sentences)
        self.build_vocab2(doc_sents)
        self.cum_table = self.make_cum_table(self.index2word, self.vocab)
        self.cum_table2 = self.make_cum_table(self.index2word2, self.vocab2)
        self.i2h = None
        self.h2o = None
        self.h2d = None
        self.init_weights(dim)
        self.neg_samples = neg_samples
        self.labels = np.zeros(neg_samples + 1)
        self.labels[0] = 1

    def train(self, iterations=5):
        sent_count = 0
        pairs = []
        for sent in self.data:
            words = [w for w in sent]
            for pos, word in enumerate(words):
                reduced_window = self.random.randint(self.window)
                start = max(0, pos - self.window + reduced_window)
                for pos2, word2 in enumerate(words[start:(pos + self.window + 1 - reduced_window)], start):
                    if pos2 != pos:
                        pairs.append((self.word2index[word], self.word2index[word2]))
            sent_count += 1
        test = collections.Counter(pairs).most_common(5000)
        print(test)
        pairs2 = []
        for sent in self.doc_data:
            doc_id = sent[0]
            for word in sent[1:]:
                pairs2.append((self.word2index[doc_id], self.word2index2[word]))

        it = len(pairs)*iterations
        if len(pairs2) > len(pairs):
            it = len(pairs2)
        epoch_size = len(pairs)
        if len(pairs2) > epoch_size:
            epoch_size = len(pairs2)
        for i in range(it):
            if i % 10000 == 0:
                print(i / 10000, ' of ', it / 10000, len(pairs), len(pairs2))
            if len(pairs) > 0:
                input, target = pairs[(i % len(pairs))]
                self.train_pair(input, target, self.cum_table, self.h2o)
            if len(pairs2) > 0:
                input2, target2 = pairs2[(i % len(pairs2))]
                self.train_pair(input2, target2, self.cum_table2, self.h2d)
            if i == epoch_size:
                self.alpha -= 0.006

    def train_pair(self, input, target, cum_table, h2o):
        l1 = self.i2h[input]
        err = np.zeros(l1.shape)

        word_indices = [target]
        while len(word_indices) < self.neg_samples + 1:
            w = cum_table.searchsorted(random.randint(0,cum_table[-1]))
            if w != target:
                word_indices.append(w)
        l2 = h2o[word_indices]
        fb = expit(np.dot(l1, l2.T)) # hidden -> output
        gb = (self.labels - fb) * self.alpha

        h2o[word_indices] += np.outer(gb, l1) # update hidden -> output
        err += np.dot(gb, l2)
        self.i2h[input] += err # update input -> hidden

    def init_weights(self, dim):
        self.i2h = np.random.uniform(0,1,(len(self.index2word),dim)) / dim
        self.h2o = np.zeros((len(self.index2word),dim), dtype=np.float32)
        self.h2d = np.zeros((len(self.index2word2), dim), dtype=np.float32)

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = np.random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.dim) - 0.5) / self.dim

    def build_vocab(self, sents):
        words = []
        for sent in sents:
            for word in sent:
                words.append(word)
        count = collections.Counter(words).most_common()
        for word, c in count:
            self.word2index[word] = len(self.word2index)
            self.vocab[word] = {'count':c}
        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

    def build_vocab2(self, sents):
        words = []
        for sent in sents:
            if sent[0] not in self.word2index:
                self.word2index[sent[0]] = len(self.word2index)
            for word in sent[1:]:
                words.append(word)
        count = collections.Counter(words).most_common()
        for word, c in count:
            self.word2index2[word] = len(self.word2index2)
            self.vocab2[word] = {'count':c}
        self.index2word2 = dict(zip(self.word2index2.values(), self.word2index2.keys()))
        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

    def make_cum_table(self, index2word, vocab, power=0.75, domain=2**31 - 1):
        vocab_size = len(vocab)
        cum_tab = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            train_words_pow += vocab[index2word[word_index]]['count']**power
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += vocab[index2word[word_index]]['count']**power
            cum_tab[word_index] = round(cumulative / train_words_pow * domain)
        if len(cum_tab) > 0:
            assert cum_tab[-1] == domain
        return cum_tab