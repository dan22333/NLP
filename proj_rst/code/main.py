from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from svm_features import svm_extract_features
from train_data import gen_train_data
from rst_parser import parse_files

work_dir = "."

if __name__ == '__main__':
	print("training..")
	trees = preprocess(work_dir, "TRAINING", "binarized", "gold")
	[train_data, EDUS_table, vocab] = gen_train_data(trees, work_dir)
	[x_features, y_labels] = svm_extract_features(train_data, EDUS_table, vocab)

	print("evaluate..")
	preprocess(work_dir, "DEV", "dev_binarized", "dev_gold")
	parse_files(work_dir)