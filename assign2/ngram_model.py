#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv
import math

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 20000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE
    start_ind = word_to_num['<s>']
    end_ind = word_to_num['</s>']
    for s in dataset: # iterate over sentences
        pair = ()
        tri = ()
        for ind in s: # iterate over words indices
            
            tri = tri + (ind,)
            if start_ind == ind:
                pair = (ind,)
                continue

            pair = pair + (ind,)
            if len(tri) == 3:
                if trigram_counts.get(tri) == None:
                    trigram_counts[tri] = 1
                else:
                    trigram_counts[tri] += 1
                tri = tri[1:]
            if len(pair) == 2:
                if bigram_counts.get(pair) == None:
                    bigram_counts[pair] = 1
                else:
                    bigram_counts[pair] += 1
                pair = pair[1:]
            if ind != end_ind:
                if unigram_counts.get(ind) == None:
                    unigram_counts[ind] = 1
                else:
                    unigram_counts[ind] += 1
                token_count += 1
    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count

def dev_trigram():
    trigram_counts, _, _, _ = train_trigram(S_dev)

    return trigram_counts

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
     ### YOUR CODE HERE
    l = 0 # log-likelihood
    lambda3 = 1 - lambda1 - lambda2 
    start_ind = word_to_num['<s>']
    end_ind = word_to_num['</s>']
    for s in eval_dataset: # sentences iterations
        tri = ()
                
        for ind in s: # words indices iterations
            tri = tri + (ind,)
            if start_ind == ind:
                continue

            if len(tri) == 3:
                p = 0
                if bigram_counts.get(tri) != None and bigram_counts.get(tri[:2]) != None:
                    p += trigram_counts[tri] / (1.0 * bigram[tri[:2]]) * lambda1
                if bigram_counts.get(tri[1:]) != None and unigram_counts.get(ind) != None:
                    p += bigram_counts[tri[1:]] / (1.0 * unigram_counts[ind]) * lambda2
                if unigram_counts.get(ind) != None:
                    p += unigram_counts[ind] / (1.0 * train_token_count) * lambda3

                if p > 0:
                    l += math.log(p, 2)
                else:
                    print "Error: zero probablity"
    l /= len(eval_dataset)
    perplexity = 2 ** (-l)
    ### END YOUR CODE
    return perplexity

def grid_search(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count):
    """
    Running grid search to tune the linear interpolation coefficients. 
    Find the perplexity for every setting of the coefficients and the setting 
    that minimizes perplexity on the dev set.
    """
    best_perplexity = -1
    best_lambda1 = 0.0
    best_lambda2 = 1.0

    lambda1 = 0
    step = 0.1
    while lambda1 <= 1.0:
        lambda2 = 0
        while lambda2 <= (1.0 - lambda1):
            perplexity = evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
            if best_perplexity < 0 or best_perplexity > perplexity:
                best_perplexity = perplexity
                best_lambda1 = lambda1
                best_lambda2 = lambda1
            lambda2 += step
        lambda1 += step

    return best_perplexity, best_lambda1, best_lambda2

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    test_ngram()