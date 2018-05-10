from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
from submitters_details import get_details
import re
import numpy as np
import copy
import heapq

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
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
    features['prev_tag'] = prev_tag
    features['prevprev_prev_tag'] = str.format("{}_{}", prevprev_tag, prev_tag)
    ### END YOUR CODE
    return features

def extract_features(sentence, i):    
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def extract_multiple_features(features, tag_value_pairs, index_to_tag_dict):
    examples = []
    example_ind = {}
    cur_ind = 0

    for t, u in tag_value_pairs:
        example = copy.copy(features)
        prev_tag = index_to_tag_dict[u]
        prevprev_tag = index_to_tag_dict[t]
        # overwrite tag related features only
        example['prev_tag'] = prev_tag
        example['prevprev_tag'] = prevprev_tag
        example['prevprev_prev_tag'] = str.format("{}_{}", prevprev_tag, prev_tag)
        examples.append(example)
        example_ind[(t,u)] = cur_ind
        cur_ind += 1
    return examples, example_ind

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
    for sent in sents:
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])
    return examples, labels

def memm_greedy(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    # clear tags
    new_sent = [(word, None) for (word, _) in sent]
    
    for i in xrange(len(sent)):
        vec_pos = vectorize_features(vec, extract_features(new_sent, i))
        tag_ind = logreg.predict(vec_pos)[0]
        tag = index_to_tag_dict[tag_ind]
        predicted_tags[i] = tag
        new_sent[i] = new_sent[i][0], tag

    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    tag_to_index_dict = invert_dict(index_to_tag_dict)

    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    n_tags = len(index_to_tag_dict)
    pi_dict = {}
    bp_dict = {}
    beam = 10 # set the number of top best tag_i-2, tag_i-1 pairs of PI_k-1

    heap = [] # min heap
    heapq.heappush(heap, (1.0, tag_to_index_dict['*'], tag_to_index_dict['*']))

    # clear tags
    new_sent = [(word, None) for (word, _) in sent]

    best_u = 0
    best_v = 0

    for k in range(len(sent)):
        best_u = 0
        best_v = 0
        best_score = 0
        if k > 1 or k > 0:
                new_sent[k - 1] = new_sent[k - 1][0], index_to_tag_dict[u]
                if k > 1:
                    new_sent[k - 2] = new_sent[k - 2][0], index_to_tag_dict[t]

        features = extract_features(new_sent, k)
        tag_value_pairs = [(t,u) for (_,t,u) in heap]
        # extract tag related features to multiple examples
        # where each pair of tags specify an example
        examples, example_ind = extract_multiple_features(features, tag_value_pairs, index_to_tag_dict)
        vec_pos = vec.transform(examples)

        while heap:
            prev_val, t, u = heapq.heappop(heap)
            
            prob = logreg.predict_proba(vec_pos)[example_ind[(t,u)]]

            for v in range(n_tags):
                if tag_to_index_dict['*'] == v: 
                    continue 
                score = prev_val * prob[v] # PI(k-1, t, u) * q(v/t,u,w,k)
                key = (u,v) 
                val = pi_dict.get(key) if pi_dict.has_key(key) == True else 0
                if score > val:
                    pi_dict[key] = score
                    bp_dict[(k, u, v)] = t
                    if score > best_score:
                        best_u = u
                        best_v = v
                        best_score = score

        is_full = False

        for key, val in pi_dict.iteritems():
            u, v = key
            elem = (val, u, v)
            if is_full or len(heap) >= beam:
                is_full = True
                if val > heap[0][0]:
                    heapq.heappushpop(heap, elem)
            else:
                heapq.heappush(heap, elem)

        pi_dict = {}

    if len(sent) > 1:
        predicted_tags[-2] = index_to_tag_dict[best_u]
    predicted_tags[-1] = index_to_tag_dict[best_v]

    for k in reversed(range(2, len(sent))):
        t = bp_dict.get((k, best_u, best_v))
        predicted_tags[k - 2] = index_to_tag_dict[t]
        best_v = best_u
        best_u = t
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

    greedy_match = 0
    vit_match = 0
    total = 0

    for i, sen in enumerate(test_data):
        predicted_tags = memm_viterbi(sen, logreg, vec, index_to_tag_dict)
        vit_match += sum([predicted_tags[j] == sen[j][1] for j in range(len(sen))])
        predicted_tags = memm_greedy(sen, logreg, vec, index_to_tag_dict)
        greedy_match += sum([predicted_tags[j] == sen[j][1] for j in range(len(sen))])
        total += len(sen)
        acc_greedy = float(greedy_match) / total   
        acc_viterbi = float(vit_match) / total

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
    all_examples_vectorized = vec.fit_transform(train_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    # print train_examples_vectorized.shape
    # print len(vec.get_feature_names())

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
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