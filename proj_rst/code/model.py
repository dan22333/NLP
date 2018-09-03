from sklearn import svm
from sklearn import linear_model
import numpy as np

from features import extract_features
from relations_inventory import ind_to_action_map

svm_max_iter = 3000
svm_verbose = False

def mini_batch_linear_model(trees, sample_ind_to_tree, samples, y_all, EDUS_table, vocab, \
	wordVectors, max_edus, iterations=100, subset_size=500, print_every=10):

	print("n_samples = {} , vocab size = {} , n_classes = {}".\
		format(len(samples), len(vocab), len(y_all)))

	classes = y_all

	clf = linear_model.SGDClassifier(tol=1e-7, learning_rate='constant', eta0=0.1)
	print(clf)

	for i in range(iterations):
		if i > 0 and i % print_every == 0:
			print("mini batch iter = {}".format(i))

		[x_vecs, y_labels] = extract_features(trees, sample_ind_to_tree, samples, \
			EDUS_table, vocab, wordVectors, subset_size, max_edus)

		linear_train(clf, x_vecs, y_labels, classes)
		classes = None

	for i in range(iterations):
		[x_vecs, y_labels] = extract_features(trees, sample_ind_to_tree, samples, \
			EDUS_table, vocab, wordVectors, subset_size, max_edus)

		dec = linear_train(clf, x_vecs, y_labels, classes)
		scores = [y_all[np.argmax(elem)] for elem in dec]
		n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		print("num matches = {}".format(n_match / len(scores) * 100))
		pred = linear_predict(clf, x_vecs)
		# print("pred = {}".format(pred))
		print(pred == scores)

	return clf

def linear_train(clf, x_vecs, y_labels, classes):
	clf.partial_fit(x_vecs, y_labels, classes)
	dec = clf.decision_function(x_vecs)
	# print(dec.shape)
	return dec

def linear_predict(clf, x_vecs):
	pred = clf.predict(x_vecs)
	# print(pred.shape)
	return pred

def non_linear_model(trees, sample_ind_to_tree, samples, EDUS_table, vocab, \
	wordVectors, max_edus, iterations=1000, subset_size=5000, print_every=10):

	print("n_samples = {} , vocab size = {}".format(len(samples), len(vocab)))

	# clf_lin = svm.LinearSVC(verbose=svm_verbose, max_iter=svm_max_iter)
	clf = svm.SVC(verbose=svm_verbose, max_iter=svm_max_iter, 
		decision_function_shape='ovr')

	print(clf)

	for i in range(iterations):
		if i > 0 and i % print_every == 0:
			print("mini batch iter = {}".format(i))

		[x_vecs, y_labels] = extract_features(trees, sample_ind_to_tree, samples, \
			EDUS_table, vocab, wordVectors, subset_size, max_edus)
		dec = non_linear_train(clf, x_vecs, y_labels)
		classes = np.unique(y_labels)
		scores = [classes[np.argmax(elem)] for elem in dec]
		n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		print("num matches = {}".format(n_match / len(scores) * 100))

def non_linear_train(clf, x_vecs, y_labels):
	# print("n_features = {}".format(len(x_vecs[0])))
	clf.fit(x_vecs, y_labels)
	if False and hasattr(clf, "n_support_"):
		arr = clf.n_support_
		ind = np.argmax(arr)
		print("{} {} {}".format(np.sum(arr), ind_to_action_map[y_labels[np.argmax(arr)]], np.max(arr)))
		print("count {}".format(y_labels.count(y_labels[ind])))

	dec = clf.decision_function(x_vecs)
	# print(dec.shape)
	return dec

