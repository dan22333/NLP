from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map

import glob
import copy
import numpy as np

print_vocab = False

last_words_in_sent_dict = {}

class Sample(object):
	def __init__(self):
		self._state = [] # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
		self._action = ''

	def print_info(self):
		print("sample {} {}".format(self._state, self._action))

def gen_train_data(trees, path, print_data=True):
	[vocab, EDUS_table] = gen_vocabulary(path)

	if print_vocab:
		for key, val in vocab.items():
			print("{} {}".format(key, val))

	samples = []
	sample_ind_to_tree = []
	offset = 0
	max_edus = 0

	for tree in trees:
		fn = path 
		fn += "\\TRAINING\\"
		fn += tree._fname
		fn += ".out.edus"
		root = tree._root
		tree._offset = offset

		num_edus = 0
		with open(fn) as fh:
			for line in fh:
				num_edus += 1

		stack = []
		tree_samples = []
		queue = [] # queue of EDUS indices
		for j in range(num_edus):
			queue.append(j + 1)

		queue = queue[::-1]
		gen_train_data_tree(root, stack, queue, tree_samples, offset)
		tree._samples = copy.copy(tree_samples)
		offset += num_edus

		if num_edus > max_edus:
			max_edus = num_edus

		if print_data:
			outfn = path
			outfn += "\\train_data\\"
			outfn += tree._fname
			with open(outfn, "w") as ofh:
				for sample in tree_samples:
					ofh.write("{} {}\n".format(sample._state, sample._action))
					samples.append(sample)
					sample_ind_to_tree.append(tree)

	y_all = [action_to_ind_map[samples[i]._action] for i in range(len(samples))]
	y_all = np.unique(y_all)

	wordVectors = loadWordVectors(vocab)

	return [samples, y_all, EDUS_table, sample_ind_to_tree, vocab, wordVectors, max_edus]
					
def gen_train_data_tree(node, stack, queue, samples, offset):
	# node.print_info()
	sample = Sample()
	if node._type == "leaf":
		sample._action = "SHIFT"
		sample._state = gen_state(stack, queue, offset)
		assert(queue.pop(-1) == node._span[0])
		stack.append(node)
	else:
		[l, r] = node._childs
		gen_train_data_tree(l, stack, queue, samples, offset)
		gen_train_data_tree(r, stack, queue, samples, offset)
		if r._nuclearity == "Satellite":
			sample._action = gen_action(node, r)
		else:
			sample._action = gen_action(node, l)
	
		sample._state = gen_state(stack, queue, offset)
		assert(stack.pop(-1) == node._childs[1])
		assert(stack.pop(-1) == node._childs[0])
		stack.append(node)

	if node._type != "Root":
		samples.append(sample)

def gen_action(parent, child):
	action = "REDUCE-"
	nuc = "NN"
	if child._nuclearity == "Satellite":
		nuc = "SN" if parent._childs[0] == child else "NS"
	action += nuc
	action += "-"
	action += map_to_cluster(child._relation)
	return action
		
def gen_state(stack, queue, offset):
	ind1 = 0
	ind2 = 0
	ind3 = 0;
	if len(queue) > 0:
		ind3 = offset + queue[-1]

	if len(stack) > 0:
		ind1 = offset + get_nuclear_edu_ind(stack[-1]) # right son
		if len(stack) > 1:
			ind2 = offset + get_nuclear_edu_ind(stack[-2]) # left son

	return [ind1, ind2, ind3]

def get_nuclear_edu_ind(node):
	if node._type == "leaf":
		return node._span[0]
	l = node._childs[0]
	r = node._childs[1]
	if l._nuclearity == "Nucleus":
		return get_nuclear_edu_ind(l)
	return get_nuclear_edu_ind(r)

def vocab_get(vocab, word):
	return vocab.get(word.lower())

def vocab_set(vocab, word, ind):
	vocab[word.lower()] = ind

def gen_vocabulary(base_path):
	vocab = {}
	EDUS_table = ['']
	last_words_in_sent = {}
	ind = 0

	path_to_out = base_path
	path_to_out += "\\TRAINING\\*.out"

	for fn in glob.glob(path_to_out):
		with open(fn) as fh:
			for sent in fh:
				sent = sent.strip()
				if sent == '':
					continue
				sent = sent.split()
				for word in sent:
					last = word == sent[-1]
					if last:
						last_words_in_sent_dict[word] = 1

	num_edus = 0
	path_to_edus = base_path
	path_to_edus += "\\TRAINING\\*.edus"

	for fn in glob.glob(path_to_edus):
		# print("fn = {}".format(fn)) 
		with open(fn) as fh:
			for edu in fh:
				edu = edu.strip()
				EDUS_table.append(edu)
				# print("edu {}".format(edu))
				edu = edu.split()
				# print("edu aft split = {}".format(edu))
				for word in edu:
					# print("word = {}".format(word))
					last = last_words_in_sent_dict.get(word)
					elems = break_to_word_elems(word, last)
					# print("elem = {}".format(elems))
					for elem in elems: 
						if not vocab_get(vocab, elem):
							vocab_set(vocab, elem, ind)
							ind += 1

	return [vocab, EDUS_table]

def split_edu_to_tokens(vocab, EDUS_table, edu_ind):
	edu = EDUS_table[edu_ind]
	edu = edu.split()
	tokens = []
	for word in edu:
		last = last_words_in_sent_dict.get(word) != None
		elems = break_to_word_elems(word, last)
		for elem in elems: 
			ind = vocab_get(vocab, elem)
			assert(ind != None)
			tokens.append(elem)
	return tokens

def gen_bag_of_words(vocab, EDUS_table, edu_ind):
	zeros = []
	for i in range(len(vocab)):
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

