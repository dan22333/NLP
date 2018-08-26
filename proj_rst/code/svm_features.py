from train_data import Sample
from train_data import gen_bag_of_words
from relations_inventory import action_to_ind_map

import random

def svm_extract_features(samples, EDUS_table, vocab, train_subset_size=500):

	x_features = [] # vectorized features kist
	y_labels = []
	
	print("n_examples = {} , n_features = {}".format(len(samples), len(vocab)))
	for i in range(train_subset_size):
		sample_ind = random.randint(0, len(samples))
		vec_concat = []
		for edu_ind in samples[sample_ind]._state:
			vec_concat.append(gen_bag_of_words(vocab, EDUS_table, edu_ind))
		x_features.append(vec_concat)
		act_ind = action_to_ind_map[samples[sample_ind]._action]
		y_labels.append(act_ind)
	return [x_features, y_labels]

		
