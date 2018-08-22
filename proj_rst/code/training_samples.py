from binarization import Node
from binarization import binarize_file
from utils import map_to_cluster
from rst_parser import Queue

import glob

path_to_dis = "*.out.dis"

class Operation(object):
	def __init__(self):
		self._state = [] # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
		self._action = ''

	def print_info(self):
		print("operation {} {}".format(self._state, self._action))

def gen_training_samples(path):
	offset = 0
	for fn in glob.glob(path):
		root = binarize_file(fn)

		base_name = fn.split('.')[0]
		edus_fn = base_name
		edus_fn += ".out.edus"
		EDUS = Queue.read_file(edus_fn)

		stack = []
		samples = []
		queue = [] # queue of EDUS indices
		for i in range(EDUS.len()):
			queue.append(i + 1)
		queue = queue[::-1]
		
		gen_training_samples_tree(root, stack, queue, samples, offset)
		offset += EDUS.len()

		outfn = "train_samples\\"
		outfn += base_name
		with open(outfn, "w") as ofh:
			for sample in samples:
				ofh.write("{} {}\n".format(sample._state, sample._action))

def gen_training_samples_tree(node, stack, queue, samples, offset):
	# node.print_info()
	operation = Operation()
	if node._type == "leaf":
		operation._action = "SHIFT"
		operation._state = gen_state(stack, queue, offset)
		assert(queue.pop(-1) == node._span[0])
		stack.append(node)
	else:
		[l, r] = node._childs
		gen_training_samples_tree(l, stack, queue, samples, offset)
		gen_training_samples_tree(r, stack, queue, samples, offset)
		if r._nuclearity == "Satellite":
			operation._action = gen_action(node, r)
		else:
			operation._action = gen_action(node, l)
	
		operation._state = gen_state(stack, queue, offset)
		assert(stack.pop(-1) == node._childs[1])
		assert(stack.pop(-1) == node._childs[0])
		stack.append(node)

	if node._type != "Root":
		samples.append(operation)

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

if __name__ == '__main__':
	gen_training_samples(path_to_dis)