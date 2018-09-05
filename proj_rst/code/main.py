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

work_dir = "."

def train_model(argv, trees, samples, y_all, vocab, wordVectors, max_edus):
	model_name = "neural"
	if len(argv) > 2:
		if argv[1] == "-m":
			model_name = argv[2]

	if model_name == "neural":
		model = neural_network_model(trees, samples, vocab, wordVectors, max_edus)
	else:
		model = mini_batch_linear_model(trees, samples, y_all, vocab, \
			wordVectors, max_edus)

	return [model_name, model]
	
if __name__ == '__main__':
	print("preprocessing")
	[trees, max_edus] = preprocess(work_dir, "TRAINING", "binarized", "gold")
	[vocab, wordVectors] = gen_vocabulary(trees, work_dir)

	print("training..")
	[samples, y_all] = gen_train_data(trees, work_dir)

	[model_name, model] = train_model(sys.argv, trees, samples, y_all, \
		vocab, wordVectors, max_edus)

	print("evaluate..")
	[dev_trees, _] = preprocess(work_dir, "DEV", "dev_binarized", "dev_gold")

	parse_files(work_dir, model_name, model, dev_trees, vocab, wordVectors, \
		max_edus, y_all)