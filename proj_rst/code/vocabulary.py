from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map
from preprocess import build_file_name

import re
import glob
import copy
import numpy as np
import nltk

class Vocab(object):
	def __init__(self):
		self._tokens = {'': 0} 
		self._wordVectors = []

def gen_vocabulary(trees, base_path, files_dir="TRAINING", print_vocab=True):
	vocab = Vocab()

	word_ind = 1
	for tree in trees:
		for edu in tree._EDUS_table:
			# print("edu {}".format(edu))
			edu = nltk.word_tokenize(edu)
			# print("edu tokens = {}".format(edu))
			for word in edu:
				if not vocab_get(vocab, word):
					vocab_set(vocab, word, word_ind)
					word_ind += 1

	vocab._wordVectors = loadWordVectors(vocab._tokens)

	if print_vocab:
		n_founds = 0
		for key, val in vocab._tokens.items():
			found = False
			if list(vocab._wordVectors[val]).count(0) < len(vocab._wordVectors[val]):
				found = True
				n_founds += 1
			# print("key = {} ind = {} in dict = {}".format(key, val, found))

		# print("words in dictionary {}%".format(n_founds / len(vocab._tokens) * 100))

	[tag_to_ind_map, _] = build_tags_dict(trees)

	return [vocab, tag_to_ind_map]

def split_edu_to_tokens(tree, edu_ind):
	word_tag_list = tree._edu_word_tag_table[edu_ind]
	return [word for word, _ in word_tag_list]

def split_edu_to_tags(tree, edu_ind):
	word_tag_list = tree._edu_word_tag_table[edu_ind]
	return [tag for _, tag in word_tag_list]

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

def build_tags_dict(trees):
	tag_to_ind_map = {'': 0}
	tag_ind = 1

	for tree in trees:
		for word_tag_list in tree._edu_word_tag_table[1:]:
			for _, tag in word_tag_list:
				if tag_to_ind_map.get(tag, None) == None:
					tag_to_ind_map[tag] = tag_ind
					tag_ind += 1

	ind_to_tag_map = [''] * len(tag_to_ind_map)	
	for tag, ind in tag_to_ind_map.items():
		ind_to_tag_map[ind] = tag

	return tag_to_ind_map, ind_to_tag_map

def vocab_get(vocab, word, use_def_word=False, def_word=''):
	val = vocab._tokens.get(word.lower())
	if val != None or not use_def_word:
		return val

	return vocab._tokens.get(def_word)

def vocab_set(vocab, word, ind):
	vocab._tokens[word.lower()] = ind
