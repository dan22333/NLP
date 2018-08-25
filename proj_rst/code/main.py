from preprocess import preprocess
from preprocess import Node
from train_data import Sample
from svm_features import svm_extract_features
from train_data import gen_train_data

if __name__ == '__main__':
	trees = preprocess("*.out.dis")
	[train_data, EDUS_table, vocab] = gen_train_data(trees, "*.out.edus", "*.out")

	[x_features, y_outs] = svm_extract_features(trees, train_data, EDUS_table, vocab)