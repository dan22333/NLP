from train_data import Sample
from relations_inventory import action_to_ind_map

def svm_extract_features(trees, train_data, EDUS_table, vects, vocab):
	x_features = []
	y_outs = []
	for i in range(len(train_data)):
		vec_concat = []
		for i in train_data[i]._state:
			vec_concat.append(vects[i])

		x_features.append(vec_concat)
		act_ind = action_to_ind_map[train_data[i]._action]
		y_outs.append(act_ind)
	return [x_features, y_outs]

		
