from train_data import Sample
from relations_inventory import action_to_ind_map
from vocabulary import split_edu_to_tags
from vocabulary import split_edu_to_tokens
from vocabulary import vocab_get
from vocabulary import DEFAULT_TOKEN # empty string
from vocabulary import get_tag_ind

import random

def extract_features(trees, samples, vocab, subset_size, max_edus, tag_to_ind_map):
	x_vecs = []
	y_labels = []

	for i in range(subset_size):
		sample_ind = random.randint(0, len(samples) - 1)
		_, vec_feats = add_features_per_sample(samples[sample_ind], vocab, max_edus, tag_to_ind_map)
		x_vecs.append(vec_feats)
		y_labels.append(action_to_ind_map[samples[sample_ind]._action])

	return [x_vecs, y_labels]

def add_features_per_sample(sample, vocab, max_edus, tag_to_ind_map, use_def=False):
	features = {}
	feat_names = []
	split_edus = []
	tags_edus = []
	tree = sample._tree
	for i in range(len(sample._state)):
		edu_ind = sample._state[i]
		if edu_ind > 0:
			split_edus.append(split_edu_to_tokens(tree, edu_ind))
			tags_edus.append(split_edu_to_tags(tree, edu_ind))
		else:
 			split_edus.append([''])
 			tags_edus.append([''])

	feat_names.append(['BEG-WORD-STACK1', 'BEG-WORD-STACK2', 'BEG-WORD-QUEUE1'])
	feat_names.append(['SEC-WORD-STACK1', 'SEC-WORD-STACK2', 'SEC-WORD-QUEUE1'])
	feat_names.append(['THIR-WORD-STACK1', 'THIR-WORD-STACK2', 'THIR-WORD-QUEUE1'])

	feat_names.append(['BEG-TAG-STACK1', 'BEG-TAG-STACK2', 'BEG-TAG-QUEUE1'])
	feat_names.append(['SEC-TAG-STACK1', 'SEC-TAG-STACK2', 'SEC-TAG-QUEUE1'])
	feat_names.append(['THIR-TAG-STACK1', 'THIR-TAG-STACK2', 'THIR-TAG-QUEUE1'])

	for i in range(0,3):
		add_word_features(features, split_edus, feat_names[i], i)

	for i in range(0,3):
		add_tag_features(features, tags_edus, feat_names[i + 3], i, tag_to_ind_map)

	feat_names = ['END-WORD-STACK1', 'END-WORD-STACK2', 'END-WORD-QUEUE1']
	add_word_features(features, split_edus, feat_names, -1)

	feat_names = ['END-TAG-STACK1', 'END-TAG-STACK2', 'END-TAG-QUEUE1']
	add_tag_features(features, tags_edus, feat_names, -1, tag_to_ind_map)

	add_edu_features(features, tree, sample._state, split_edus, max_edus)

	vecs = gen_vectorized_features(features, vocab, tag_to_ind_map, use_def)
	return features, vecs

def add_word_features(features, split_edus, feat_names, word_loc):
	for i in range(len(split_edus)):
		words = split_edus[i]
		feat = feat_names[i]
		features[feat] = DEFAULT_TOKEN
		if words != ['']:
			# last word or one of the first 3 words
			if word_loc < 0 or len(words) > word_loc:
				features[feat] = words[word_loc]

def add_tag_features(features, tags_edus, feat_names, tag_loc, tag_to_ind_map):
	for i in range(len(tags_edus)):
		tags = tags_edus[i]
		feat = feat_names[i]
		features[feat] = ''
		if tags != ['']:
			if tag_loc < 0 or len(tags) > tag_loc:
				features[feat] = tags[tag_loc]

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
			edu_ind_in_tree.append(edus_ind[i]) 
		else:
			edu_ind_in_tree.append(0)

	features['DIST-FROM-START-QUEUE1'] = (edu_ind_in_tree[2] - 1.0) / max_edus

	features['DIST-FROM-END-STACK1'] = \
		(tree._root._span[1] - edu_ind_in_tree[0]) / max_edus

	features['DIST-STACK1-QUEUE1'] = \
		(edu_ind_in_tree[2] - edu_ind_in_tree[0]) / max_edus 

	same_sent = tree._edu_to_sent_ind[edus_ind[0]] == \
		tree._edu_to_sent_ind[edus_ind[2]]

	features['SAME-SENT-STACK1-QUEUE1'] = 1 if same_sent else 0

def gen_vectorized_features(features, vocab, tag_to_ind_map, use_def):
	vecs = []
	n_tags = len(tag_to_ind_map) - 1
	for key, val in features.items():
		# print("key {} val {}".format(key, val))
		if 'WORD' in key:
			word_ind = vocab_get(vocab, val, use_def)
			vecs += [elem for elem in vocab._wordVectors[word_ind]]
		elif 'TAG' in key:
			vecs += [get_tag_ind(tag_to_ind_map, val, use_def) / n_tags]
		else:
			vecs += [val]
		# print(len(vecs))
	return vecs
