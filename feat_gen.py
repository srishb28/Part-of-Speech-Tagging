#!/bin/python
import nltk
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
import os
os.environ['JAVAHOME'] = "C:\Program Files\Java\jdk-11.0.1"
nltk.download('punkt')
from nltk.tag.stanford import StanfordNERTagger
import random
import itertools
import logging
from math import log, isnan, isinf
from collections import defaultdict

clustdict = {}
#def make_float_defaultdict():
#    return defaultdict(float)
#
#
#def make_int_defaultdict():
#    return defaultdict(int)
#class ClassLMClusters(object):
#    '''
#    The initializer takes a document generator, which is simply an iterator
#    over lists of tokens.  You can define this however you wish.
#    '''
#def __init__(self, corpus_path, batch_size=1000, max_vocab_size=None,lower=False):
#
#        self.batch_size = batch_size
#        self.corpus_path = corpus_path
#        self.lower = lower  # whether to lowercase everything
#
#        self.max_vocab_size = max_vocab_size
#
#        # mapping from cluster IDs to cluster IDs,
#        # to keep track of the hierarchy
#        self.cluster_parents = {}
#        self.cluster_counter = 0
#
#        # the list of words in the vocabulary and their counts
#        self.counts = defaultdict(int)
#        self.trans = defaultdict(make_int_defaultdict)
#        self.num_tokens = 0
#
#        # the graph weights (w) and the effects of merging nodes (L)
#        # (see Liang's thesis)
#        self.w = defaultdict(make_float_defaultdict)
#        self.L = defaultdict(make_float_defaultdict)
#
#        # the 0/1 bit to add when walking up the hierarchy
#        # from a word to the top-level cluster
#        self.cluster_bits = {}
#
#        # find the most frequent words
#        self.vocab = {}
#        self.reverse_vocab = []
#        self.create_vocab()
#
#        # create sets of documents that each word appears in
#        self.create_index()
#
#        logging.getLogger().setLevel(logging.INFO)
#        # make a copy of the list of words, as a queue for making new clusters
#        word_queue = list(range(len(self.vocab)))
#
#        # score potential clusters, starting with the most frequent words.
#        # also, remove the batch from the queue
#        self.current_batch = word_queue[:(self.batch_size + 1)]
#        word_queue = word_queue[(self.batch_size + 1):]
#        self.initialize_tables()
#
#        while len(self.current_batch) > 1:
#            # find the best pair of words/clusters to merge
#            c1, c2 = self.find_best()
#
#            # merge the clusters in the index
#            self.merge(c1, c2)
#
#            if word_queue:
#                new_word = word_queue.pop(0)
#                self.add_to_batch(new_word)
#
#            logging.info('{} AND {} WERE MERGED INTO {}. {} REMAIN.'
#                         .format(self.reverse_vocab[c1] if c1 < len(self.reverse_vocab) else c1,
#                                 self.reverse_vocab[c2] if c2 < len(self.reverse_vocab) else c2,
#                                 self.cluster_counter,
#                                 len(self.current_batch) + len(word_queue) - 1))
#
#            self.cluster_counter += 1
#
#def corpus_generator(self):
#        i = 0
#        for line in self.corpus_path:
#            i += 1
#            if i % 100000 == 0:
#                logging.info('Read {} nonblank lines'.format(i))
#            for tok in line:
#                if self.lower:
#                    tok = tok.lower()
#                yield tok
#
#def create_index(self):
#        corpus_iter1, corpus_iter2 = itertools.tee(self.corpus_generator())
#
#        # increment one iterator to get consecutive tokens
#        next(corpus_iter2)
#
#        for w1, w2 in zip(corpus_iter1, corpus_iter2):
#            if w1 in self.vocab and w2 in self.vocab:
#                self.trans[self.vocab[w1]][self.vocab[w2]] += 1
#
#        logging.info('{} word tokens were processed.'.format(self.num_tokens))
#
#def create_vocab(self):
#        tmp_counts = defaultdict(int)
#        for w in self.corpus_generator():
#            tmp_counts[w] += 1
#            self.num_tokens += 1
#
#        words = sorted(tmp_counts.keys(), key=lambda w: tmp_counts[w],
#                       reverse=True)
#
#        too_rare = 0
#        if self.max_vocab_size is not None \
#                and len(words) > self.max_vocab_size:
#            too_rare = tmp_counts[words[self.max_vocab_size]]
#            if too_rare == tmp_counts[words[0]]:
#                too_rare += 1
#                logging.info("max_vocab_size too low.  Using all words that" +
#                             " appeared > {} times.".format(too_rare))
#
#        for i, w in enumerate(w for w in words if tmp_counts[w] > too_rare):
#            self.vocab[w] = i
#            self.counts[self.vocab[w]] = tmp_counts[w]
#
#        self.reverse_vocab = sorted(self.vocab.keys(),
#                                    key=lambda w: self.vocab[w])
#        self.cluster_counter = len(self.vocab)
#
#def initialize_tables(self):
#        logging.info("initializing tables")
#
#        # edges between nodes
#        for c1, c2 in itertools.combinations(self.current_batch, 2):
#            w = self.compute_weight([c1], [c2]) \
#                + self.compute_weight([c2], [c1])
#            if w:
#                self.w[c1][c2] = w
#
#        # edges to and from a single node
#        for c in self.current_batch:
#            w = self.compute_weight([c], [c])
#            if w:
#                self.w[c][c] = w
#
#        num_pairs = 0
#        for c1, c2 in itertools.combinations(self.current_batch, 2):
#            self.compute_L(c1, c2)
#            num_pairs += 1
#            if num_pairs % 1000 == 0:
#                logging.info("{} pairs precomputed".format(num_pairs))
#
#def compute_weight(self, nodes1, nodes2):
#        paircount = 0
#        for n1 in nodes1:
#            for n2 in nodes2:
#                paircount += self.trans[n1][n2]
#
#        if not paircount:
#            # TODO is there some better option than returning 0 (indicating no weight)?
#            # Otherwise, it would return 0 * infinity...
#            return 0.0
#
#        count_1 = 0
#        count_2 = 0
#        for n in nodes1:
#            count_1 += self.counts[n]
#        for n in nodes2:
#            count_2 += self.counts[n]
#
#        # convert to floats
#        num_tokens = float(self.num_tokens)
#        paircount = float(paircount)
#        count_1 = float(count_1)
#        count_2 = float(count_2)
#
#        return (paircount / num_tokens) \
#               * log(paircount * num_tokens / count_1 / count_2)
#
#def compute_L(self, c1, c2):
#        val = 0.0
#
#        # add the weight of edges coming in to the potential
#        # new cluster from other nodes
#        # TODO this is slow
#        for d in self.current_batch:
#            val += self.compute_weight([c1, c2], [d])
#            val += self.compute_weight([d], [c1, c2])
#
#        # ... but don't include what will be part of the new cluster
#        for d in [c1, c2]:
#            val -= self.compute_weight([c1, c2], [d])
#            val -= self.compute_weight([d], [c1, c2])
#
#        # add the weight of the edge from the potential new cluster
#        # to itself
#        val += self.compute_weight([c1, c2], [c1, c2])
#
#        # subtract the weight of edges to/from c1, c2
#        # (which would be removed)
#        for d in self.current_batch:
#            for c in [c1, c2]:
#                if d in self.w[c]:
#                    val -= self.w[c][d]
#                elif c in self.w[d]:
#                    val -= self.w[d][c]
#
#        self.L[c1][c2] = val
#
#def find_best(self):
#        best_score = float('-inf')
#        argmax = None
#
#        for c1 in self.L:
#            for c2, score in self.L[c1].items():
#                if score > best_score:
#                    argmax = [(c1, c2)]
#                    best_score = score
#                elif score == best_score:
#                    argmax.append((c1, c2))
#
#        if isnan(best_score) or isinf(best_score):
#            raise ValueError("bad value for score: {}".format(best_score))
#
#        # break ties randomly (randint takes inclusive args!)
#        c1, c2 = argmax[random.randint(0, len(argmax) - 1)]
#
#        return c1, c2
#
#def merge(self, c1, c2):
#        c_new = self.cluster_counter
#
#        # record parents
#        self.cluster_parents[c1] = c_new
#        self.cluster_parents[c2] = c_new
#        r = random.randint(0, 1)
#        self.cluster_bits[c1] = str(r)  # assign bits randomly
#        self.cluster_bits[c2] = str(1 - r)
#
#        # add the new cluster to the counts and transitions dictionaries
#        self.counts[c_new] = self.counts[c1] + self.counts[c2]
#        for c in [c1, c2]:
#            for d, val in self.trans[c].items():
#                if d == c1 or d == c2:
#                    d = c_new
#                self.trans[c_new][d] += val
#
#        # subtract the weights for the merged nodes from the score table
#        # TODO this is slow
#        for c in [c1, c2]:
#            for d1 in self.L:
#                for d2 in self.L[d1]:
#                    self.L[d1][d2] -= self.compute_weight([d1, d2], [c])
#                    self.L[d1][d2] -= self.compute_weight([c], [d1, d2])
#
#        # remove merged clusters from the counts and transitions dictionaries
#        # to save memory (but keep frequencies for words for the final output)
#        if c1 >= len(self.vocab):
#            del self.counts[c1]
#        if c2 >= len(self.vocab):
#            del self.counts[c2]
#
#        del self.trans[c1]
#        del self.trans[c2]
#        for d in self.trans:
#            for c in [c1, c2]:
#                if c in self.trans[d]:
#                    del self.trans[d][c]
#
#        # remove the old clusters from the w and L tables
#        for table in [self.w, self.L]:
#            for d in table:
#                if c1 in table[d]:
#                    del table[d][c1]
#                if c2 in table[d]:
#                    del table[d][c2]
#            if c1 in table:
#                del table[c1]
#            if c2 in table:
#                del table[c2]
#
#        # remove the merged items
#        self.current_batch.remove(c1)
#        self.current_batch.remove(c2)
#
#        # add the new cluster to the w and L tables
#        self.add_to_batch(c_new)
#
#def add_to_batch(self, c_new):
#        # compute weights for edges connected to the new node
#        for d in self.current_batch:
#            self.w[d][c_new] = self.compute_weight([d], [c_new])
#            self.w[d][c_new] = self.compute_weight([c_new], [d])
#        self.w[c_new][c_new] = self.compute_weight([c_new], [c_new])
#
#        # add the weights from this new node to the merge score table
#        # TODO this is slow
#        for d1 in self.L:
#            for d2 in self.L[d1]:
#                self.L[d1][d2] += self.compute_weight([d1, d2], [c_new])
#                self.L[d1][d2] += self.compute_weight([c_new], [d1, d2])
#
#        # compute scores for merging it with all clusters in the current batch
#        for d in self.current_batch:
#            self.compute_L(d, c_new)
#
#        # now add it to the batch
#        self.current_batch.append(c_new)
#
#def get_bitstring(self, w):
#        # walk up the cluster hierarchy until there is no parent cluster
#        cur_cluster = self.vocab[w]
#        bitstring = ""
#        while cur_cluster in self.cluster_parents:
#            bitstring = self.cluster_bits[cur_cluster] + bitstring
#            cur_cluster = self.cluster_parents[cur_cluster]
#        return bitstring
#
#def save_clusters(self, output_path):
#        with open(output_path, 'w') as f:
#            for w in self.vocab:
#                # convert the counts back to ints when printing
#                f.write("{}\t{}\t{}\n".format(w, self.get_bitstring(w),
#                                              self.counts[self.vocab[w]]))
#        global n
#        for w in self.vocab:
#            clustdict[w] = self.get_bitstring(w)
#            if len(clustdict[w]) < n:
#                n = len(clustdict[w])

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
#    global st
#    nretagsdict = defaultdict(list)
#    st = StanfordNERTagger(
#        'C:/Users/Srishti/Desktop/nlp/cse538-assignment-2 (2018-Fall)_V2/Assignment2_for_students/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
#        'C:/Users/Srishti/Desktop/nlp/cse538-assignment-2 (2018-Fall)_V2/Assignment2_for_students/stanford-ner-2018-02-27/stanford-ner.jar',
#        encoding='utf-8')
#
#    for train_sent in train_sents:
#        sentence = construct_sentence(train_sent).decode('utf-8').replace(" \'", "\'")
#        tokenized_text = word_tokenize(sentence)
#        classified_text = st.tag(tokenized_text)
#        nretagsdict[sentence] = classified_text
#    with open('nertags.pickle', 'wb') as handle:
#        pickle.dump(nretagsdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    global n
    n = 35
    with open("exout.txt") as f:
       content = f.readlines()
    for x in content:
       c = x.split()
       clustdict[c[0]] = c[1]
#       #wordfreq[c[0]] = c[2]
   
    pass
#def construct_sentence(sent):
#        sentence = ''
#        for word in sent:
#            sentence = sentence + word.strip('\\\\') + ' '
#        return sentence
def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    stop_words = set(stopwords.words('english'))
    emoticons =r"""
    (?:
        [:=;] #Eyes
        [oO\-]? # Nose
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
    capitals_inside=word[1:].lower() != word[1:] 
    emoticon_re = re.compile(r'^'+emoticons+'$', re.VERBOSE | re.IGNORECASE)
    if sent[i] in clustdict:
       #for j in xrange(1, n):
           if  len(clustdict[sent[i]]) >= n:
               ftrs.append(clustdict[sent[i]][:n])
##    # some features of the word
#
    sno = nltk.stem.SnowballStemmer('english')
    ftrs.append(sno.stem(word))
    lemmatizer = WordNetLemmatizer()
    ftrs.append(lemmatizer.lemmatize(word))
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word[0]=="#":
        ftrs.append("HASHTAG")
    if word[0]=="@":
        ftrs.append("TAG")
    if word[0].isupper():
        ftrs.append("first_letter_CAPITALIZED_noun")
    if "-" in word:
        ftrs.append("HAS_HYPHEN")
    if word in stop_words:
        ftrs.append("STOP_WORDS")
    ftrs.append("PREFIX-1:"+word[0])
    ftrs.append("SUFFIX-1:"+word[-1])
    if len(word)>1 :
        ftrs.append("PREFIX-2:"+word[:2])
        ftrs.append("SUFFIX-2:"+word[-2:])
    if len(word)>2:
        ftrs.append("PREFIX-3:"+word[:3])
        ftrs.append("SUFFIX-3:"+word[-3:])
    ftrs.append("NO OF CHARACTERS:" + str(len(word)))
    if 'http' in word or '.com' in word:
        ftrs.append("LINK")
    if capitals_inside:
        ftrs.append("CAPITALS INSIDE")
   
    for  p in word:
        if p in string.punctuation:
            ftrs.append("punctuation")
    if emoticon_re.search(word):
        ftrs.append("EMOTICON")
#    with open('nertags.pickle', 'rb') as handle:
#       nertagsdict = pickle.load(handle)
#    sentence = construct_sentence(sent).decode('utf-8').replace(" \'", "\'")
#    if nertagsdict.has_key(sentence):
#       classified_text = nertagsdict[sentence]
#       ner_tag_dict = {}
#       for tag in classified_text:
#           ner_tag_dict[tag[0]] = tag[1]
#       if (ner_tag_dict.has_key(word)) & (ner_tag_dict.get(word) != 'O'):
#           ftrs.append("NER_TAG_" + ner_tag_dict[word])    

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
