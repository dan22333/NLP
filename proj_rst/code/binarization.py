import re

class Node(object):
	def __init__(self):
		self._nuclearity = '' # 'Nucleus' ֻ| 'Satellite'
		self._relation = ''
		self._childs = []
		self._type = '' # 'span' | 'leaf' | 'Root'
		self._span = [0,0]
		self._text = ''
		self._leaf = 0

def binarize_tree_rec(fh, stack):
	line = fh.readline()
	line = line.strip()
	words = line.split()

	if words[0] == "(":
		node = Node()

		# ( Root (span 1 54)
		m = re.match("\( Root \(span (\d+) (\d+)\)", line)
		if m:
			tokens = m.groups()
			node._nuclearity = "Root"
			node._span = [int(tokens[0]), int(tokens[1])]
			stack.append(node)
			return binarize_tree_rec(fh, stack)

		# ( Nucleus (span 1 34) (rel2par Topic-Drift)
		m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par (\w+|\w+\-\w+)\)", line)
		if m:
			tokens = m.groups()
			node._nuclearity = tokens[0]
			node._type = "span"
			node._span = [int(tokens[1]), int(tokens[2])]
			node._relation = tokens[3]
			parent = stack[-1]
			parent._childs.append(node)
			stack.append(node)
			return binarize_tree_rec(fh, stack)

		# ( Satellite (leaf 3) (rel2par attribution) (text _!Southern Co. 's Gulf Power Co. unit_!) )
		m = re.match("\( (\w+) \(leaf (\d+) \) \(rel2par (\w+|\w+\-\w+)\)", line)
		if m:
			tokens = m.groups()
			node._type = "leaf"
			node._nuclearity = tokens[0]
			node._leaf = int(tokens[1])
			node._relation = tokens[2]
			node._text = parse_text(words)
			parent = stack[-1]
			parent._childs.append(node)
			return node
	else:
		return stack.pop(-1)

def binarize_tree(filaname):
	stack = []
	with open(filename) as fh:
		root = binarize_tree_rec(fh, stack)
	return root


# (text _!Southern Co. 's Gulf Power Co. unit may plead_!)
def parse_text(words):
	for i in range(len(words)):
		if word == "(text":
			break

	text = words[i + 1]
	text = text[2:]
	return text[:-3]


			


