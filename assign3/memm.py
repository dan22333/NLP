from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
from submitters_details import get_details
import re


def extract_features_base_aa(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    ### YOUR CODE HERE
    features['word'] = curr_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word
    features['next_word'] = next_word
    for i in xrange(1, 5):
        if len(curr_word) > i:
            features[str.format("pref{}", i)] = curr_word[:i]
            features[str.format("suff{}", i)] = curr_word[len(curr_word) - i:]
    features['prevprev_tag'] = prevprev_tag
    features['bigram'] = prev_tag
    features['trigram'] = str.format("{}_{}", prevprev_tag, prev_tag)
    ### END YOUR CODE
    return features

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['tag_bigram'] = prevprev_tag + "-" + prev_tag
    features['prev_wordtag_pairs'] = prev_word + "/" + prev_tag
    features['prevprev_wordtag_pairs'] = prevprev_word + "/" + prevprev_tag
    features['next_word'] = next_word

    if curr_word in vocab and vocab[curr_word] >= MIN_FREQ:
        features['pref1'] = curr_word[0]
        features['pref2'] = curr_word[0:2]
        features['pref3'] = curr_word[0:3]
        features['pref4'] = curr_word[0:4]
        features['suf1'] = curr_word[-1]
        features['suf2'] = curr_word[-2:]
        features['suf3'] = curr_word[-3:]
        features['suf4'] = curr_word[-4:]

        if re.compile(".*[A-Z]").match(curr_word):
            features['has_uppercase'] = 1
        if re.compile(".*\d").match(curr_word):
            features['has_number'] = 1
        if re.compile(".*-").match(curr_word):
            features['has_hyphen'] = 1

    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    count = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])
    return examples, labels


def memm_greeedy(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return predicted_tags

def should_add_eval_log(sentene_index):
    if sentene_index > 0 and sentene_index % 10 == 0:
        if sentene_index < 150 or sentene_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    for i, sen in enumerate(test_data):

        ### YOUR CODE HERE
        ### Make sure to update Viterbi and greedy accuracy
        raise NotImplementedError
        ### END YOUR CODE

        if should_add_eval_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i), str(acc_greedy), str(acc_viterbi) , str (eval_end_timer - eval_start_timer))
            eval_start_timer = time.time()

    return str(acc_viterbi), str(acc_greedy)

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    print all_examples_vectorized.shape
    print len(vec.get_feature_names())

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=250, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "End training, elapsed " + str(end - start) + " seconds"
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print "Start evaluation on dev set"
    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec, index_to_tag_dict)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"