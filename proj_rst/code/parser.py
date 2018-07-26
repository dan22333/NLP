import glob
import random

from relations_inventory import relations_list
from relations_inventory import is_multi_nuclear_relation

path_to_trainig = "dataset\\TRAINING\\*.out.edus"

class Node(object):
	def __init__(self):
		self._nuclearity = 'N' # nuclearity status - either nucleus (N) or satellite (S) 
		self._relation = 'span' # relation to parent
		self._left = ''
		self._right = ''
		self._EDU = ''

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
				queue._EDUS += line
				# print("{}".format(line))
		return queue

	def empty(self):
		return self._EDUS == []

	def pop(self):
		return self._EDUS.pop(0)

class Transition(object):
	def __init__(self):
		self._nuclearity = '' # either 'N' or 'S'
		self._relation = ''
		self._action = '' # shift or 'reduce'

def parse_files(root):
	path = root
	path += "\\"
	path += path_to_trainig

	for filename in glob.glob(path):
		queue = Queue.read_file(filename)
		print("{}".format(filename))
		stack = Stack()
		parse(queue, stack)

def get_transition(queue, stack):
	transition = Transition()

	if stack.size() < 2:
		transition._action = 'shift'
	elif not queue.empty():
		actions = ['shift', 'reduce']
		ind = random.randint(0,1)
		transition._action = actions[ind]
	else:
		transition._action = 'reduce'
		
	if transition._action == 'shift':
		return transition

	nuclearity_options = ['N','S']
	transition._nuclearity = nuclearity_options[random.randint(0,1)]
	transition._relation = relations_list[random.randint(0, len(relations_list) - 1)]

	return transition

def invert_nuclearity(nuclearity):
	if nuclearity == 'N':
		return 'S'
	return 'N'

def parse(queue, stack):
	while not queue.empty() or stack.size() != 1:
		node = Node()
		transition = get_transition(queue, stack)
		if transition._action == 'shift':
			# create a leaf
			node._EDU = queue.pop() 
		else:
			node._left = stack.pop()
			node._right = stack.pop()
			node._relation = transition._relation

			if not is_multi_nuclear_relation(node._relation):
				node._left._nuclearity = transition._nuclearity
				node._right._nuclearity = invert_nuclearity(transition._nuclearity)
		stack.push(node)

	return []

if __name__ == '__main__':
	parse_files("..")
