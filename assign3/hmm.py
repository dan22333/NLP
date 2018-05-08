from data import *
import time
from submitters_details import get_details
from tester import verify_hmm_model


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE
    for i in range(len(sents)):
        first = "*";
        second = "*"
        for word,tag in sents[i]:
            total_tokens +=1
            if word not in q_uni_counts:
                q_uni_counts[tag] = 1
            else:
                q_uni_counts[word] +=1
            if (second,tag) not in q_bi_counts:
                 q_bi_counts[(second,tag)] =1
            else:
                 q_bi_counts[(second,tag)] +=1
            if (first,second,tag) not in q_tri_counts:
                q_tri_counts[(first,second,tag)] =1
            else:
                q_tri_counts[(first,second,tag)] +=1
            if (word,tag) not in e_word_tag_counts:
                 e_word_tag_counts[(word,tag)] =1
            else:
                 e_word_tag_counts[(word,tag)] +=1
            if tag not in e_tag_counts:
                e_tag_counts[tag] =1
            else:
                e_tag_counts[tag] +=1
            first = second
            second = tag

    q_bi_counts[("*","*")] = total_tokens
    q_uni_counts["*"] = total_tokens
    for key in q_tri_counts:
        q_tri_counts[key] =  q_tri_counts[key]*1.0 /q_bi_counts[(key[0],key[1])]
    for key in q_bi_counts:
        q_bi_counts[key] = q_bi_counts[key]*1.0/ q_uni_counts[key[0]]
    for key in q_uni_counts:
        q_uni_counts[key] =  q_uni_counts[key]*1.0/ total_tokens
    for key in e_word_tag_counts:
        e_word_tag_counts[key] = e_word_tag_counts[key]*1.0/  e_tag_counts[key[1]]
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return str(acc_viterbi)

if __name__ == "__main__":
    print (get_details())
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    verify_hmm_model(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "Dev: Accuracy of Viterbi hmm: " + acc_viterbi

    train_dev_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "Test: Accuracy of Viterbi hmm: " + acc_viterbi
        full_flow_end = time.time()
        print "Full flow elapsed: " + str(full_flow_end - start_time) + " seconds"