import re
import copy
import filecmp
import glob

from utils import map_to_cluster

dis_files_dir = "*.out.dis"

class Node(object):
	def __init__(self):
		self._nuclearity = '' # 'Nucleus' | 'Satellite'
		self._relation = ''
		self._childs = []
		self._type = '' # 'span' | 'leaf' | 'Root'
		self._span = [0, 0]
		self._text = ''

	def copy(self):
		to = Node()
		to._nuclearity = self._nuclearity
		to._relation = self._relation
		to._childs = copy.copy(self._childs)
		to._span = copy.copy(self._span)
		to._text = self._text
		to._type = self._type
		return to

	def print_info(self):
		node_type = self._type
		beg = self._span[0]
		end = self._span[1]
		nuc = self._nuclearity
		rel = self._relation
		text = self._text
		print("node: {} {} {} {} {} {}".format(node_type, beg, end, nuc, rel, text))

def binarize_files(path):
	for fn in glob.glob(path):
		binarize_file(fn)

def binarize_file(infn):
	stack = []
	with open(infn, "r") as ifh:
		lines = ifh.readlines()
		root = build_tree(lines[::-1], stack)

	binarize_tree(root)

	outfn = infn
	outfn = "binarized\\"
	outfn += infn
	with open(outfn, "w") as ofh:
		print_dis(ofh, root, 0)

	# res = filecmp.cmp(infn, outfn)
	# print("compare files {} {} same = {}".format(infn, outfn, res))

	goldfn = "gold\\"
	goldfn += infn.split('.')[0]

	with open(goldfn, "w") as ofh:
		print_gold(ofh, root)

def build_tree(lines, stack):
	line = lines.pop(-1)
	line = line.strip()

	# print("{}".format(line))
 
	node = Node()

	# ( Root (span 1 54)
	m = re.match("\( Root \(span (\d+) (\d+)\)", line)
	if m:
		tokens = m.groups()
		node._type = "Root"
		node._span = [int(tokens[0]), int(tokens[1])]
		stack.append(node)
		return build_tree_childs_iter(lines, stack)

	# ( Nucleus (span 1 34) (rel2par Topic-Drift)
	m = re.match("\( (\w+) \(span (\d+) (\d+)\) \(rel2par ([\w-]+)\)", line)
	if m:
		tokens = m.groups()
		node._nuclearity = tokens[0]
		node._type = "span"
		node._span = [int(tokens[1]), int(tokens[2])]
		node._relation = tokens[3]
		stack.append(node)
		return build_tree_childs_iter(lines, stack)

	# ( Satellite (leaf 3) (rel2par attribution) (text _!Southern Co. 's Gulf Power Co. unit_!) )
	m = re.match("\( (\w+) \(leaf (\d+)\) \(rel2par ([\w-]+)\) \(text (.+)", line)
	tokens = m.groups()
	node._type = "leaf"
	node._nuclearity = tokens[0]
	node._span = [int(tokens[1]), int(tokens[1])] 
	node._relation = tokens[2]
	text = tokens[3]
	text = text[2:]
	text = text[:-5]
	node._text = text
	# node.print_info()
	return node
	
def build_tree_childs_iter(lines, stack):
	# stack[-1].print_info()

	while True:
		line = lines[-1]
		line.strip()
		words = line.split()
		if words[0] == ")":
			lines.pop(-1)
			break

		node = build_tree(lines, stack)
		stack[-1]._childs.append(node)
	return stack.pop(-1)

def binarize_tree(node):
	if node._childs == []:
		return

	if len(node._childs) > 2:
		stack = []
		for child in node._childs:
			stack.append(child)

		node._childs = []
		while len(stack) > 2:
			# print("deree > 2")
			r = stack.pop(-1)
			l = stack.pop(-1)

			t = l.copy()
			t._childs = [l, r]
			t._span = [l._span[0], r._span[1]]
			t._type = "span"
			stack.append(t)
		r = stack.pop(-1)
		l = stack.pop(-1)
		node._childs = [l, r]
	else:
		l = node._childs[0]
		r = node._childs[1]

	binarize_tree(l)
	binarize_tree(r)

# print tree in .dis format
def print_dis(ofh, node, level):
	nuc = node._nuclearity
	rel = node._relation
	beg = node._span[0]
	end = node._span[1]
	if node._type == "leaf":
		# Nucleus (leaf 1) (rel2par span) (text _!Wall Street is just about ready to line_!) )
		print_spaces(ofh, level)
		text = node._text
		ofh.write("( {} (leaf {}) (rel2par {}) (text _!{}_!) )\n".format(nuc, beg, rel, text))
	else:
		if node._type == "Root":
			# ( Root (span 1 91)
			ofh.write("( Root (span {} {})\n".format(beg, end))
		else:
			# ( Nucleus (span 1 69) (rel2par Contrast)
			print_spaces(ofh, level)
			ofh.write("( {} (span {} {}) (rel2par {})\n".format(nuc, beg, end, rel))
		l = node._childs[0]
		r = node._childs[1]
		print_dis(ofh, l, level + 1)
		print_dis(ofh, r, level + 1) 
		print_spaces(ofh, level)
		ofh.write(")\n")

def print_spaces(ofh, level):
	n_spaces = 2 * level
	for i in range(n_spaces):
		ofh.write(" ")

def print_gold(ofh, node):
	if node._type != "Root":
		nuc = node._nuclearity
		rel = map_to_cluster(node._relation)
		beg = node._span[0]
		end = node._span[1]
		ofh.write("{} {} {} {}\n".format(beg, end, nuc[0], rel))

	if node._type != "leaf":
		l = node._childs[0]
		r = node._childs[1]
		print_gold(ofh, l)
		print_gold(ofh, r)

if __name__ == '__main__':
	# binarize_file("0600.out.dis")
	binarize_files(dis_files_dir)








			


