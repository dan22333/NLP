from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map
from preprocess import build_file_name

import glob
import copy
import numpy as np

class Vocab(object):
	def __init__(self):
		self._tokens = {} 
		self._last_words_in_sent = {}

def gen_vocabulary(trees, base_path, print_vocab=False):
	vocab = Vocab()

	for tree in trees:
		# read sentences
		fn = build_file_name(tree._fname, base_path, "TRAINING", "out") 
		with open(fn) as fh:
			for sent in fh:
				sent = sent.strip()
				if sent == '':
					continue
				sent = sent.split()
				for word in sent:
					last = word == sent[-1]
					if last:
						vocab._last_words_in_sent[word] = 1

	word_ind = 0
	for tree in trees:
		fn = build_file_name(tree._fname, base_path, "TRAINING", "out.edus")
		# print("fn = {}".format(fn)) 
		with open(fn) as fh:
			for edu in fh:
				edu = edu.strip()
				# print("edu {}".format(edu))
				edu = edu.split()
				# print("edu aft split = {}".format(edu))
				for word in edu:
					# print("word = {}".format(word))
					last = vocab._last_words_in_sent.get(word)
					elems = break_to_word_elems(word, last)
					# print("elem = {}".format(elems))
					for elem in elems: 
						if not vocab_get(vocab, elem):
							vocab_set(vocab, elem, word_ind)
							word_ind += 1

	if print_vocab:
		for key, val in vocab._tokens.items():
			print("{} {}".format(key, val))

	wordVectors = loadWordVectors(vocab._tokens)

	return [vocab, wordVectors]

def split_edu_to_tokens(vocab, EDUS_table, edu_ind, def_word='', use_def_word=False):
	edu = EDUS_table[edu_ind]
	edu = edu.split()
	tokens = []
	for word in edu:
		last = vocab._last_words_in_sent.get(word) != None
		elems = break_to_word_elems(word, last)
		for elem in elems: 
			ind = vocab_get(vocab, elem)
			if not use_def_word:
				assert(ind != None)
			if ind == None:
				print("word not in vocabulary {}".format(elem))
				elem = def_word
			tokens.append(elem)
	return tokens

def gen_bag_of_words(vocab, EDUS_table, edu_ind):
	zeros = []
	for i in range(len(vocab._tokens)):
		zeros.append(0)

	if edu_ind == 0:
		return zeros

	vec = zeros
	tokens = split_edu_to_tokens(vocab, EDUS_table, edu_ind)
	for token in tokens:
		ind = vocab_get(vocab, token)
		vec[ind] += 1
	return vec	

def break_to_word_elems(word, last):
	# strip suffices attached to the last word in sentence
	if word[-1] in ['.','!','?'] and last == True:
		elems = []
		if len(word) > 1: # ". . ."
			elems = break_to_word_elems_do(word[0:-1])
		elems.append(word[-1])
		return elems
	# strip other suffices 
	elif word[-1] in [')','"',"\'","`"]:
		elems = []
		if len(word) > 1:
			elems = break_to_word_elems_do(word[0:-1])
		elems.append(word[-1])
		return elems
	return break_to_word_elems_do(word)

def break_to_word_elems_do(word):
	elems = []
	suf = ''
	mid = word
	if mid[0] in ['"', '(', '`']:
		elems.append(word[0])
		mid = word[1:]
		if mid == '':
			return elems
	if mid[-1] in ['\'', '"',')','!','?',',',':','-']:
		suf = mid[-1]
		mid = mid[0:-1]
	if mid == "'s" or mid[-2:].lower() == "'s":
		suf = mid[-2:]
		mid = mid[0:-2]

	if mid == '':
		elems.append(suf)
		return elems

	if mid[-1] != '.':
		elems.append(mid)
		if suf != '':
			elems.append(suf)
		return elems

	# Abbreviation - Etc.
	if mid[0].isupper() and mid[1:-1].islower() and mid[-1] == '.':
		elems.append(mid)
	# Person name initial - M.
	elif mid[0].isupper() and mid[-1] == '.':
		elems.append(mid)
	# Acronym - i.e.
	elif mid.islower() and mid[-1] == '.' and mid[:-1].find('.') > 0:
		elems.append(mid)		
	else: # dot is not part of the word 
		elems.append(mid[:-1])
		elems.append(mid[-1])
	
	if suf != '':
		elems.append(suf)
	return elems

def vocab_get(vocab, word, print_vocab=False):
	return vocab._tokens.get(word.lower())

def vocab_set(vocab, word, ind):
	vocab._tokens[word.lower()] = ind
