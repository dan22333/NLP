from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from train_data import gen_train_data
from rst_parser import parse_files
from model import mini_batch_linear_model 

work_dir = "."

if __name__ == '__main__':
	print("preprocessing")
	trees = preprocess(work_dir, "TRAINING", "binarized", "gold")
	print("training..")
	[samples, y_all, EDUS_table, sample_ind_to_tree, vocab, wordVectors, max_edus] = \
		gen_train_data(trees, work_dir)
	clf = mini_batch_linear_model(trees, sample_ind_to_tree, samples, y_all, \
		EDUS_table, vocab, wordVectors, max_edus)
	print("evaluate..")
	preprocess(work_dir, "DEV", "dev_binarized", "dev_gold")
	parse_files(work_dir)