from preprocess import Node
from preprocess import preprocess
from utils import map_to_cluster

import glob

class Sample(object):
	def __init__(self):
		self._state = [] # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
		self._action = ''

	def print_info(self):
		print("sample {} {}".format(self._state, self._action))

def gen_train_data(trees, path_to_edus, path_to_out):
	[vocab, EDUS_table, EDUS_bag_of_words] = gen_vocabulary(path_to_out)

	offset = 0
	EDUS = []
	i = 0
	for fn in glob.glob(path_to_edus): # "*.out.edu"
		root = trees[i]
		i += 1
		num_edus = 0
		with open(fn) as fh:
			for line in fh:
				num_edus += 1

		stack = []
		samples = []
		queue = [] # queue of EDUS indices
		for i in range(num_edus):
			queue.append(i + 1)

		queue = queue[::-1]
		
		gen_train_data_tree(root, stack, queue, samples, offset)
		offset += num_edus

		outfn = "train_samples\\"
		outfn += base_name
		with open(outfn, "w") as ofh:
			for sample in samples:
				ofh.write("{} {}\n".format(sample._state, sample._action))
				train_data.append(smaple)
	return [train_data, EDUS_table, EDUS_bag_of_words, vocab]
					
def gen_train_data_tree(node, stack, queue, samples, offset):
	# node.print_info()
	samples = Sample()
	if node._type == "leaf":
		sample._action = "SHIFT"
		sample._state = gen_state(stack, queue, offset)
		assert(queue.pop(-1) == node._span[0])
		stack.append(node)
	else:
		[l, r] = node._childs
		gen_training_samples_tree(l, stack, queue, samples, offset)
		gen_training_samples_tree(r, stack, queue, samples, offset)
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
	left_right = "LEFT" if parent._childs[0] == child else "RIGHT"
	action = "REDUCE-"
	action += map_to_cluster(child._relation)
	action += "-"
	action += left_right
	return action
		
def gen_state(stack, queue, offset):
	ind1 = 0
	ind2 = 0
	ind3 = 0;
	if len(queue) > 0:
		ind3 = offset + queue[-1]

	if len(stack) > 0:
		ind1 = offset + get_nuclear_edu_ind(stack[-1])
		if len(stack) > 1:
			ind2 = offset + get_nuclear_edu_ind(stack[-2])

	return [ind1, ind2, ind3]

def get_nuclear_edu_ind(node):
	if node._type == "leaf":
		return node._span[0]
	l = node._childs[0]
	r = node._childs[1]
	if l._nuclearity == "Nucleus":
		return get_nuclear_edu_ind(l)
	return get_nuclear_edu_ind(r)

def gen_vocabulary(path_to_out):
	vocab = {}
	EDUS_table = ['']
	EDUS_bag_of_words = []
	ind = 0

	for fn in glob.glob(path_to_out): 
		with open(fn) as fh:
			sent = fh.readline()
			sent = sent.strip()
			# print(sent)
			sent = sent.split()
			# print(sent)
			for word in sent:
				# print(word)
				last = word == sent[-1]
				elems = break_to_word_elems(word, last)
				# print(elems)
				for elem in elems: 
					if not vocab.get(elem):
						vocab[elem] = ind
						ind += 1

		base_name = fn.split('.')[0]
		edus_fn = base_name
		edus_fn += ".out.edus"
		num_edus = 0

		with open(edus_fn) as edu_fh:
			for line in edu_fh:
				line = line.strip()
				EDUS_table.append(line)

	EDUS_bag_of_words = gen_bag_of_words_edus(vocab, EDUS_table)

	return [vocab, EDUS_table, EDUS_bag_of_words]

def break_to_word_elems(word, last):
	if word[-1] in ['.','!','?'] and last == True:
		elems = break_to_word_elems_do(word[0:-1])
		elems.append(word[-1])
		return elems
	if word[-1] in [')','"']:
		elems = break_to_word_elems_do(word[0:-1])
		elems.append(word[-1])
		return elems
	return break_to_word_elems_do(word)

def break_to_word_elems_do(word):
	elems = []
	suf = ''
	mid = word
	if mid[0] in ['"', '(']:
		elems.append(word[0])
		mid = word[1:]
	if mid[-1] in ['\'', '"',')','!','?',',',':']:
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

def gen_bag_of_words_edus(vocab, EDUS_table):
	to = len(EDUS_table)
	for i in range(len(EDUS_table))
		v = gen_bag_of_words(vocab, EDUS_table, i)
		EDUS_bag_of_words.append(v)

	return EDUS_bag_of_words

def gen_bag_of_words(vocab, EDUS_table, edu_ind):
	zeros = []
	for i in range(vocab_size):
		zeros.append(0)

	if edu_ind == 0:
		return zeros

	vec = zeros
	edu = EDUS_table[edu_ind]
	edu = edu.split()
	for word in edu:
		last = word == edu[-1]
		elems = break_to_word_elems(word, last)
		for elem in elems: 
			ind = vocab.get(elem)
			assert(ind != None)
			vec[ind] += 1
	return vec	

