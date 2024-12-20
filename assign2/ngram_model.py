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
vocabsize = 2000
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
	lines = 0
	start_ind = word_to_num['<s>']
	for s in dataset: # iterate over sentences
		pair = ()
		tri = ()
		for ind in s: # iterate over words
			tri = tri + (ind,)
			if start_ind == ind:
				continue

			pair = tri[1:]

			if trigram_counts.get(tri) == None:
				trigram_counts[tri] = 1
			else:
				trigram_counts[tri] += 1

			tri = tri[1:] # shift right

			if bigram_counts.get(pair) == None:
				bigram_counts[pair] = 1
			else:
				bigram_counts[pair] += 1

			if unigram_counts.get(ind) == None:
				unigram_counts[ind] = 1
			else:
				unigram_counts[ind] += 1

			token_count += 1
		lines += 1
	
	bigram_counts[(start_ind, start_ind)] = len(dataset)
	unigram_counts[start_ind] = len(dataset)

	return trigram_counts, bigram_counts, unigram_counts, token_count

def dev_trigram():
	trigram_counts, _, _, _ = train_trigram(S_dev)

	return trigram_counts

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, 
	train_token_count, lambda1, lambda2, step=0.05):
	"""
    	Goes over an evaluation dataset and computes the perplexity for it with
    	the current counts and a linear interpolation
	"""
	perplexity = 0
	word_counts = 0
	l = 0 # log-likelihood
	lambda3 = 1.0 - lambda1 - lambda2 
	lambda3 = 0.0 if lambda3 < step / 2.0 else lambda3
	# print("lambda3 " + str(lambda3))
	start_ind = word_to_num['<s>']
	for s in eval_dataset: # sentences iterations
		tri = ()

		for ind in s: # words indices iterations
			tri = tri + (ind,)
			if start_ind == ind:
				continue

			p = 0.0 # initialize probability
			if trigram_counts.get(tri) != None and bigram_counts.get(tri[:2]) != None:
				p += lambda1 * trigram_counts[tri] / (1.0 * bigram_counts[tri[:2]])
			if bigram_counts.get(tri[1:]) != None and unigram_counts.get(tri[1]) != None:
				p += lambda2 * bigram_counts[tri[1:]] / (1.0 * unigram_counts[tri[1]])
			if unigram_counts.get(ind) != None:
				p += lambda3 * unigram_counts[ind] / (1.0 * train_token_count)

			if p > 0:
				l += np.log2([p], dtype=np.float)[0]
			else:
				l += -np.inf
			tri = tri[1:] # shift right
			word_counts += 1

	l /= (1.0 * word_counts)
	perplexity = np.exp2([-l], dtype=np.float)[0]

	return perplexity

def grid_search(eval_dataset, trigram_counts, bigram_counts, unigram_counts, 
	train_token_count):
	"""
		Running grid search to tune the linear interpolation coefficients. 
		Find the perplexity for every setting of the coefficients and the setting 
		that minimizes perplexity on the dev set.
	"""
	best_perplexity = -1
	best_lambda1 = 0
	best_lambda2 = 0

	lambda1 = 0
	step = 0.05
	while lambda1 <= 1.0:
		lambda2 = 0
		while lambda2 <= (1.0 - lambda1):
			perplexity = evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, 
				unigram_counts, train_token_count, lambda1, lambda2)
			print("#perplexity: " + str(perplexity) + " , lambda1: " 
				+ str(lambda1) + " ', lambda2: " + str(lambda2))
			if best_perplexity < 0 or best_perplexity > perplexity:
				best_perplexity = perplexity
				best_lambda1 = lambda1
				best_lambda2 = lambda2
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
	perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, 
		token_count, 0.4, 0.5)
	print "#perplexity: " + str(perplexity) + " , lambda1: 0.5 , lambda2: 0.4"
	perplexity, lambda1, lambda2 = grid_search(S_dev, trigram_counts, 
		bigram_counts, unigram_counts, token_count)
	print("#best perplexity: " + str(perplexity) + " , lambda1: " + 
		str(lambda1) + " , lambda2: " + str(lambda2))

if __name__ == "__main__":
	test_ngram()