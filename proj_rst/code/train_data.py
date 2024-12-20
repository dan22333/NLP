from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map
from vocabulary import gen_vocabulary
from preprocess import build_file_name
from preprocess import SEP

import glob
import copy
import numpy as np

class Sample(object):
	def __init__(self):
		self._state = [] # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
		self._action = ''
		self._tree = ''

	def print_info(self):
		print("sample {} {}".format(self._state, self._action))

def gen_train_data(trees, path, print_data=False):
	samples = []

	for tree in trees:
		fn = build_file_name(tree._fname, path, "TRAINING", "out.edus")
		root = tree._root
		stack = []
		tree_samples = []
		queue = [] # queue of EDUS indices
		for j in range(tree._root._span[1]):
			queue.append(j + 1)

		queue = queue[::-1]
		gen_train_data_tree(root, stack, queue, tree_samples)
		tree._samples = copy.copy(tree_samples)

		if print_data:
			outfn = path
			outfn += SEP + "train_data" + SEP
			outfn += tree._fname
			with open(outfn, "w") as ofh:
				for sample in tree_samples:
					ofh.write("{} {}\n".format(sample._state, sample._action))
		
		for sample in tree_samples:
			sample._tree = tree
			samples.append(sample)

	y_all = [action_to_ind_map[samples[i]._action] for i in range(len(samples))]
	y_all = np.unique(y_all)

	return [samples, y_all]
					
def gen_train_data_tree(node, stack, queue, samples):
	# node.print_info()
	sample = Sample()
	if node._type == "leaf":
		sample._action = "SHIFT"
		sample._state = gen_state(stack, queue)
		assert(queue.pop(-1) == node._span[0])
		stack.append(node)
	else:
		[l, r] = node._childs
		gen_train_data_tree(l, stack, queue, samples)
		gen_train_data_tree(r, stack, queue, samples)
		if r._nuclearity == "Satellite":
			sample._action = gen_action(node, r)
		else:
			sample._action = gen_action(node, l)
	
		sample._state = gen_state(stack, queue)
		assert(stack.pop(-1) == node._childs[1])
		assert(stack.pop(-1) == node._childs[0])
		stack.append(node)

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
		
def gen_state(stack, queue):
	ind1 = 0
	ind2 = 0
	ind3 = 0;
	if len(queue) > 0:
		ind3 = queue[-1]

	if len(stack) > 0:
		ind1 = get_nuclear_edu_ind(stack[-1]) # right son
		if len(stack) > 1:
			ind2 = get_nuclear_edu_ind(stack[-2]) # left son

	return [ind1, ind2, ind3]

def get_nuclear_edu_ind(node):
	if node._type == "leaf":
		return node._span[0]
	l = node._childs[0]
	r = node._childs[1]
	if l._nuclearity == "Nucleus":
		return get_nuclear_edu_ind(l)
	return get_nuclear_edu_ind(r)
