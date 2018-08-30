from preprocess import preprocess
from preprocess import Node
from preprocess import TreeInfo
from train_data import Sample
from svm_features import svm_extract_features
from train_data import gen_train_data

work_dir = "."

if __name__ == '__main__':
	trees = preprocess(work_dir)
	[train_data, EDUS_table, vocab] = gen_train_data(trees, work_dir, print_data=True)
	[x_features, y_labels] = svm_extract_features(train_data, EDUS_table, vocab)