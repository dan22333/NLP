from data import *
from submitters_details import get_details
import tester

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """

    ### YOUR CODE HERE
    dict = {}
    all_tokens = [] # flat list of tokens

    for sent in train_data:
        for token in sent:
            all_tokens.append(token)

    all_tokens.sort()
    # all_tokens.reverse()

    best_tok = all_tokens[0]
    most_freq = 1
    count = 1
    curr = best_tok

    for i in xrange(1, len(all_tokens)):
        # new word or same word with new tag
        if curr[0] != all_tokens[i][0] or curr[1] != all_tokens[i][1]:
            if count > most_freq:
                most_freq = count
                best_tok = curr
            if curr[0] != all_tokens[i][0]: # new word
                dict[best_tok[0]] = best_tok[1]
                most_freq = 1
                best_tok = all_tokens[i]
            count = 0
        curr = all_tokens[i]
        count += 1 

    if count > most_freq:
        most_freq = count
        best_tok = curr
    dict[best_tok[0]] = best_tok[1]

    return dict
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    total = 0
    match = 0
    for sent in test_set:
        for word, expected_tag in sent:
                total += 1
                tag = pred_tags.get(word, "NotFound")
                if tag == expected_tag:
                    match += 1
                    
    return str(float(match) / total)
    ### END YOUR CODE

if __name__ == "__main__":
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    tester.verify_most_frequent_model(model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)