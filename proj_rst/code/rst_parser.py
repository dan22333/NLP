import glob
import random

from preprocess import Node
from preprocess import print_serial_file
from preprocess import extract_base_name_file
from evaluation import eval

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
		self._nuclearity = [] # <nuc>, <nuc>
		self._relation = '' # cluster relation
		self._action = '' # shift or 'reduce'

def parse_files(base_path, edus_files_dir="DEV", gold_files_dir="dev_gold"):
	path = base_path
	path += "\\"
	path += edus_files_dir
	path += "\\*.out.edus"

	for fn in glob.glob(path):
		queue = Queue.read_file(fn)
		stack = Stack()
		root = parse_file(queue, stack)

		predfn = base_path
		predfn += "\\pred\\"
		base_name = extract_base_name_file(fn)
		predfn += base_name
		with open(predfn, "w") as ofh:
			print_serial_file(ofh, root, False)

	eval(gold_files_dir, "pred")
		# n1 = count_lines(predfn) 
		# n2 = count_lines(goldfn)
		# print("{} {} {} {} equal: {}".format(predfn, n1, goldfn, n2, n1 == n2))

def parse_file(queue, stack):
	leaf_ind = 1
	while not queue.empty() or stack.size() != 1:
		node = Node()
		node._relation = 'SPAN'
		transition = most_freq_baseline(queue, stack)
		if transition._action == "shift":
			# create a leaf
			node._text = queue.pop()
			node._type = 'leaf'
			node._span = [leaf_ind, leaf_ind]
			leaf_ind += 1
		else:
			r = stack.pop()
			l = stack.pop()
			node._childs.append(l)
			node._childs.append(r)
			l._nuclearity = transition._nuclearity[0]
			r._nuclearity = transition._nuclearity[1]
			if l._nuclearity == "Satellite":
				l._relation = transition._relation
			elif r._nuclearity == "Satellite":
				r._relation = transition._relation	
			else:
				l._relation = transition._relation
				r._relation = transition._relation

			if queue.empty() and stack.size() == 0:
				node._type = "Root"
			else:
				node._type = "span"
			node._span = [l._span[0], r._span[1]]
		stack.push(node)

	return stack.pop()

def count_lines(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines

def most_freq_baseline(queue, stack):
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

if __name__ == '__main__':
	parse_files("..")
