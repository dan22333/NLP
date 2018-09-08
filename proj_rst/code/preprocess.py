import re
import copy
import filecmp
import glob
import nltk
from nltk import tokenize
from nltk import pos_tag
import os

from utils import map_to_cluster
from relations_inventory import build_parser_action_to_ind_mapping

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
		print("node: type= {} span = {},{} nuc={} rel={} text={}".\
			format(node_type, beg, end, nuc, rel, text))

class TreeInfo(object):
	def __init__(self):
		self._fname = '' # file name
		self._root = ''
		self._EDUS_table = ['']
		self._sents = ['']
		self._edu_to_sent_ind = ['']
		self._edu_word_tag_table = [['']]

def preprocess(path, dis_files_dir, bin_files_dir, ser_files_dir):
	build_parser_action_to_ind_mapping()

	[trees, max_edus] = binarize_files(path, dis_files_dir, bin_files_dir)
	print_serial_files(path, trees, ser_files_dir)

	gen_sentences(trees, path, dis_files_dir)

	# statistics
	num_edus = 0
	match_edus = 0

	for tree in trees:
		# print("file {} ".format(tree._fname))
		sent_ind = 1
		n_sents = len(tree._sents)
		fn = build_file_name(tree._fname, path, dis_files_dir, "out.edus")
		# print("fn = {}".format(fn)) 
		with open(fn) as fh:
			for edu in fh:
				edu = edu.strip()
				edu_tokenized = tokenize.word_tokenize(edu)
				tree._edu_word_tag_table.append(nltk.pos_tag(edu_tokenized))
				tree._EDUS_table.append(edu)
				if not edu in tree._sents[sent_ind]:
					sent_ind += 1
				tree._edu_to_sent_ind.append(sent_ind)
				if edu in tree._sents[sent_ind]:
					match_edus += 1
				num_edus += 1
				# print("edu = {}".format(edu))
				# print("{} {}".format(sent_ind, tree._sents[sent_ind]))
			assert(sent_ind < n_sents)

	print("num match between edu and a sentence {} , num edus {} , {}%".\
		format(match_edus, num_edus, match_edus / num_edus * 100.0))

	return [trees, max_edus]

def binarize_files(base_path, dis_files_dir, bin_files_dir):
	trees = []
	max_edus = 0
	path = base_path
	path += "\\"
	path += dis_files_dir

	path += "\\*.dis"
	for fn in glob.glob(path):
		tree = binarize_file(fn, bin_files_dir)
		trees.append(tree)
		if tree._root._span[1] > max_edus:
			max_edus = tree._root._span[1]
	return [trees, max_edus]

# return the root of the binarized file

def binarize_file(infn, bin_files_dir):
	stack = []
	with open(infn, "r") as ifh: # .dis file
		lines = ifh.readlines()
		root = build_tree(lines[::-1], stack)

	binarize_tree(root)

	outfn = infn.split("\\")[0]
	outfn += "\\"
	outfn += bin_files_dir
	outfn += "\\"
	outfn += extract_base_name_file(infn)
	outfn += ".out.dis"
	with open(outfn, "w") as ofh:
		print_dis_file(ofh, root, 0)

	# res = filecmp.cmp(infn, outfn)
	# print("compare files {} {} same = {}".format(infn, outfn, res))

	tree_info = TreeInfo()
	tree_info._root = root
	tree_info._fname = extract_base_name_file(infn)
	return tree_info

def extract_base_name_file(fn):
	base_name = fn.split("\\")[-1]
	base_name = base_name.split('.')[0]
	return base_name

# lines are the content of "out.dis" file

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
def print_dis_file(ofh, node, level):
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
		print_dis_file(ofh, l, level + 1)
		print_dis_file(ofh, r, level + 1) 
		print_spaces(ofh, level)
		ofh.write(")\n")

def print_spaces(ofh, level):
	n_spaces = 2 * level
	for i in range(n_spaces):
		ofh.write(" ")

# print serial tree files

def print_serial_files(base_path, trees, outdir):
	remove_dir(base_path, outdir)
	path = base_path
	path += "\\"
	path += outdir

	os.makedirs(path)

	for tree in trees:
		outfn = path
		outfn += "\\"
		outfn += tree._fname
		with open(outfn, "w") as ofh:
			print_serial_file(ofh, tree._root)

def print_serial_file(ofh, node, doMap=True):
	if node._type != "Root":
		nuc = node._nuclearity
		if doMap == True:
			rel = map_to_cluster(node._relation)
		else:
			rel = node._relation
		beg = node._span[0]
		end = node._span[1]
		ofh.write("{} {} {} {}\n".format(beg, end, nuc[0], rel))

	if node._type != "leaf":
		l = node._childs[0]
		r = node._childs[1]
		print_serial_file(ofh, l, doMap)
		print_serial_file(ofh, r, doMap)

def gen_sentences(trees, base_path, infiles_dir):
	sents_dir = "sents"

	if not os.path.isdir(sents_dir):
   		os.makedirs(sents_dir)

	for tree in trees:
		fn = tree._fname
		# print("file = {}".format(tree._fname))
		fn = build_file_name(tree._fname, base_path, infiles_dir, "out") 
		fn_sents = build_file_name(tree._fname, base_path, "sents", "out.sents")
		with open(fn) as fh:
			# read the text
			content = ''
			lines = fh.readlines()
			for line in lines:
				if line.strip() != '':
					content += line
			# print("content {}".format(content))
			# break the text into sentences
			sents = tokenize.sent_tokenize(content)
			with open(fn_sents, "w") as ofh:
				for sent in sents:
					# print(sent)
					sent = sent.replace('\n', ' ')
					sent = sent.replace('  ', ' ')
					if sent.strip() == "\.":
						continue
					ofh.write("{}\n".format(sent))	
					tree._sents.append(sent)
	
def build_file_name(base_fn, base_path, files_dir, suf):
	fn = base_path
	fn += "\\"
	fn += files_dir
	fn += "\\"
	fn += base_fn
	fn += "."
	fn += suf
	return fn

def remove_dir(base_path, dir):
	path = base_path
	path += "\\"
	path += dir
	if os.path.isdir(dir):
		path_to_files = path
		path_to_files += "\\*"
		for fn in glob.glob(path_to_files):
			os.remove(fn)
		os.rmdir(path)

if __name__ == '__main__':
	# binarize_file("0600.out.dis")
	binarize_files(dis_files_dir)








			


