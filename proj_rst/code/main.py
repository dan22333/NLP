from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from train_data import gen_train_data
from rst_parser import parse_files
from model import mini_batch_linear_model 
from model import neural_network_model
from vocabulary import gen_vocabulary

import sys

WORK_DIR = "."
TRAINING_DIR = "TRAINING"
DEV_DIR = "DEV"

def train_model(argv, trees, samples, y_all, vocab, max_edus, tag_to_ind_map):
	model_name = "neural"
	if len(argv) > 2:
		if argv[1] == "-m":
			model_name = argv[2]

	if model_name == "neural":
		model = neural_network_model(trees, samples, vocab, max_edus, tag_to_ind_map)
	else:
		model = mini_batch_linear_model(trees, samples, y_all, vocab, \
			max_edus, tag_to_ind_map)

	return [model_name, model]
	
if __name__ == '__main__':
	print("preprocessing")
	[trees, max_edus] = preprocess(WORK_DIR, TRAINING_DIR, "binarized", "gold")
	[vocab, tag_to_ind_map] = gen_vocabulary(trees, WORK_DIR, TRAINING_DIR)

	print("training..")
	[samples, y_all] = gen_train_data(trees, WORK_DIR)

	[model_name, model] = train_model(sys.argv, trees, samples, y_all, \
		vocab, max_edus, tag_to_ind_map)

	print("evaluate..")
	[dev_trees, _] = preprocess(WORK_DIR, DEV_DIR, "dev_binarized", "dev_gold")

	parse_files(WORK_DIR, "dev_gold", model_name, model, dev_trees, vocab, \
		max_edus, y_all, tag_to_ind_map, DEV_DIR)