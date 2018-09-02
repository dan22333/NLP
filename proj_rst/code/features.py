from train_data import Sample
from relations_inventory import action_to_ind_map
from train_data import split_edu_to_tokens
from train_data import vocab_get

import random

def extract_features(trees, sample_ind_to_tree, samples, EDUS_table, vocab, wordVectors, subset_size, max_edus):
	x_vecs = []
	y_labels = []
	# text_labels = []

	for i in range(subset_size):
		sample_ind = random.randint(0, len(samples) - 1)
		tree = sample_ind_to_tree[sample_ind]
		_, vec_feats = add_features_per_sample(samples[sample_ind], tree, EDUS_table, vocab, \
			wordVectors, max_edus)
		x_vecs.append(vec_feats)
		y_labels.append(action_to_ind_map[samples[sample_ind]._action])
		# text_labels.append(samples[sample_ind]._action)

	return [x_vecs, y_labels]

def add_features_per_sample(sample, tree, EDUS_table, vocab, wordVectors, max_edus):
	features = {}
	feat_names = []
	split_edus = []
	for i in range(len(sample._state)):
		edu_ind = sample._state[i]
		if edu_ind > 0:
 			split_edus.append(split_edu_to_tokens(vocab, EDUS_table, edu_ind))
		else:
 			split_edus.append([''])

	feat_names.append(['BEG-WORD-STACK1', 'BEG-WORD-STACK2', 'BEG-WORD-QUEUE1'])
	feat_names.append(['SEC-WORD-STACK1', 'SEC-WORD-STACK2', 'SEC-WORD-QUEUE1'])
	feat_names.append(['THIR-WORD-STACK1', 'THIR-WORD-STACK2', 'THIR-WORD-QUEUE1'])

	for i in range(0,3):
		add_word_features(features, split_edus, feat_names[i], i)

	feat_names = ['END-WORD-STACK1', 'END-WORD-STACK2', 'END-WORD-QUEUE1']
	add_word_features(features, split_edus, feat_names, -1)

	add_edu_features(features, tree, sample._state, split_edus, max_edus)

	vecs = gen_vectorized_features(features, vocab, wordVectors)
	return features, vecs

def add_edu_features(features, tree, edus_ind, split_edus, max_edus):
	feat_names = ['LEN-STACK1', 'LEN-STACK2', 'LEN-QUEUE1']

	for i in range(0,3):
		feat = feat_names[i]
		if edus_ind[i] > 0:
			features[feat] = len(split_edus[i]) / max_edus
		else:
			features[feat] = 0

	edu_ind_in_tree = []

	for i in range(0,3):
		if edus_ind[i] > 0:
			edu_ind_in_tree.append(edus_ind[i] - tree._offset) 
		else:
			edu_ind_in_tree.append(0)

	features['DIST-FROM-START-QUEUE1'] = (edu_ind_in_tree[2] - 1.0) / max_edus

	features['DIST-FROM-END-STACK1'] = \
		(tree._root._span[1] - edu_ind_in_tree[0]) / max_edus

	features['DIST-STACK1-QUEUE1'] = \
		(edu_ind_in_tree[2] - edu_ind_in_tree[0]) / max_edus 

def add_word_features(features, split_edus, feat_names, word_ind):
	for i in range(len(split_edus)):
		words = split_edus[i]
		feat = feat_names[i]
		features[feat] = ''
		if words != '':
			if word_ind < 0:
				features[feat] = words[word_ind]
			elif len(words) > word_ind:
				features[feat] = words[word_ind]

def gen_vectorized_features(features, vocab, wordVectors):
	vecs = []
	for key, val in features.items():
		if 'WORD' in key:
			word_ind = vocab_get(vocab, val)
			vecs += [elem for elem in wordVectors[word_ind]]
		else:
			vecs += [val]
	return vecs
