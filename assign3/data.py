import os
MIN_FREQ = 3
def invert_dict(d):
	res = {}
	for k, v in d.iteritems():
		res[v] = k
	return res

def read_conll_pos_file(path):
	"""
		Takes a path to a file and returns a list of word/tag pairs
	"""
	sents = []
	with open(path, "r") as f:
		curr = []
		for line in f:
			line = line.strip()
			if line == "":
				sents.append(curr)
				curr = []
			else:
				tokens = line.strip().split("\t")
				curr.append((tokens[1],tokens[3]))
	return sents

def increment_count(count_dict, key):
	"""
		Puts the key in the dictionary if does not exist or adds one if it does.
		Args:
			count_dict: a dictionary mapping a string to an integer
			key: a string
	"""
	if key in count_dict:
		count_dict[key] += 1
	else:
		count_dict[key] = 1

def compute_vocab_count(sents):
	"""
		Takes a corpus and computes all words and the number of times they appear
	"""
	vocab = {}
	for sent in sents:
		for token in sent:
			increment_count(vocab, token[0])
	return vocab

def hasDigit(str):
	for c in str:
		if c.isdigit() == True:
			return True
	return False

def hasAlpha(str):
	for c in str:
		if c.isalpha() == True:
			return True
	return False

def hasChar(str, ch):
	for c in str:
		if c == ch:
			return True
	return False

def replace_word(word, firstWord):
	"""
		Replaces rare words with categories (numbers, dates, etc...)
	"""
	### YOUR CODE HERE

	# Two/Four digit year 90, 1990
	if len(word) == 2 or len(word) == 4: 
		if word.isdigit() == True:
			if len(word) == 2:
				return "twoDigitNum"
			return "fourDigitNum"

	# Product code (A8056-67)
	if hasDigit(word) == True and hasAlpha(word) == True:
		return "containsDigitAndAlpha"

	# Date 09-96
	if hasDigit(word) == True and hasChar(word, '-') == True:
		return "containsDigitAndDash"

	# Date 11/9/89
	if hasDigit(word) == True and hasChar(word, '/') == True:
		return "containsDigitAndSlash"

	# Time 22:15
	if hasDigit(word) == True and hasChar(word, ':') == True:
		return "containsDigitAndColon"

	# Monetary anount 23,000.00
	if hasDigit(word) == True and hasChar(word, ',') == True:
		return "containsDigitAndComma"

	# Other number 1.00
	if hasDigit(word) == True and hasChar(word, '.') == True:
		return "containsDigitAndPeriod"

	# Other number 456789
	if word.isdigit() == True:
		return "othernum"

	# Organization BBN
	if word.isupper() and word.isalpha() == True:
		return "allCaps"

	# Abbreviation - Etc.
	if word[0].isupper() and word[1:-1].islower() and word[-1] == '.':
		return "abbreviationWord"

	# Person name initial - M.
	if word[0].isupper() and word[-1] == '.':
		return "capPeriod"

	# Acronym - i.e.
	if word.islower() and word[-1] == '.':
		return "acronymWord"

	# first word of sentence - no useful capitalization information
	if firstWord == True:
		return "firstWord"

	# Sally - capitalized word
	if word[0].isupper() == True:
		return "initCap"

	# uncapitalized word
	if word.islower() == True:
		return "lowercase"

	# Punctuation marks, all other words
	return "otherWord"
	
	### END YOUR CODE

def preprocess_sent(vocab, sents):
	"""
		return a sentence, where every word that is not frequent enough is replaced
	"""
	res = []
	total, replaced = 0, 0

	for sent in sents:
		new_sent = []
		firstWord = True
		for token in sent:
			if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
				new_sent.append(token)
			else:
				new_sent.append((replace_word(token[0], firstWord), token[1]))
				replaced += 1 
			total += 1
			firstWord = False
		res.append(new_sent)
	print "replaced: " + str(float(replaced)/total)
	return res
