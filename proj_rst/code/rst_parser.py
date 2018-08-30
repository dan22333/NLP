import glob
import random

from relations_inventory import cluster_rels_list
from binarization import Node
from binarization import print_gold

path_to_training = "dataset\\TRAINING\\*.out.edus"

class Stack(object):
	def __init__(self):
		self._stack = []

	def pop(self):
		return self._stack.pop(-1)

	def push(self, elem):
		return self._stack.append(elem)

	def size(self):
		return len(self._stack)

class Queue(object):
	def __init__(self):
		self._EDUS = []

	@classmethod
	def read_file(cls, filename):
		# print("{}".format(filename))
		queue = Queue()
		with open(filename) as fh:
			for line in fh:
				line = line.strip()
				queue._EDUS.append(line)
			queue._EDUS[::-1]
		return queue

	def empty(self):
		return self._EDUS == []

	def pop(self):
		return self._EDUS.pop(-1)

	def len(self):
		return len(self._EDUS)

class Transition(object):
	def __init__(self):
		self._nuclearity = [] # either 'Nucleus' or 'Satellite' to left and right nodes
		self._relation = '' # cluster relation
		self._action = '' # shift or 'reduce'

def parse_files(root):
	path = root
	path += "\\"
	path += path_to_training

	for filename in glob.glob(path):
		queue = Queue.read_file(filename)
		print("{}".format(filename))
		stack = Stack()
		root = parse(queue, stack)

		predfn = "pred\\"
		base_name = filename.split('\\')[-1]
		base_name = base_name.split('.')[0]
		predfn += base_name
		with open(predfn, "w") as ofh:
			print_gold(ofh, root, False)

		goldfn = "gold\\"
		goldfn += base_name

		# n1 = count_lines(predfn) 
		# n2 = count_lines(goldfn)
		# print("{} {} {} {} equal: {}".format(predfn, n1, goldfn, n2, n1 == n2))
		
def count_lines(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines

def gen_most_freq_baseline(queue, stack):
	transition = Transition()

	if stack.size() < 2:
		transition._action = "shift"
	elif not queue.empty():
		actions = ["shift", "reduce"]
		ind = random.randint(0,1)
		transition._action = actions[ind]
	else:
		transition._action = "reduce"
		
	if transition._action == "shift":
		return transition

	transition._relation = 'ELABORATION'
	transition._nuclearity.append("Nucleus")
	transition._nuclearity.append("Satellite")

	return transition

def parse(queue, stack):
	leaf_ind = 1
	while not queue.empty() or stack.size() != 1:
		node = Node()
		transition = get_transition(queue, stack)
		if transition._action == "shift":
			# create a leaf
			node._text = queue.pop()
			node._type = 'leaf'
			node._span = [leaf_ind, leaf_ind]
			leaf_ind += 1
		else:
			childs = [stack.pop(), stack.pop()]
			if childs[0]._span[0] > childs[1]._span[1]:
				childs = childs[::-1]
			[l, r] = childs
			node._childs.append(l)
			node._childs.append(r)
			node._relation = transition._relation
			l._nuclearity = transition._nuclearity[0]
			r._nuclearity = transition._nuclearity[1]
			if queue.empty() and stack.size() == 0:
				node._type = "Root"
			else:
				node._type = "span"
			node._span = [l._span[0], r._span[1]]
		stack.push(node)

	return stack.pop()

if __name__ == '__main__':
	parse_files("..")
