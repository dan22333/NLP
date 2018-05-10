from data import *
import time
from submitters_details import get_details
from tester import verify_hmm_model
from collections import defaultdict, deque
import numpy as np

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
            if tag not in q_uni_counts:
                q_uni_counts[tag] = 1
            else:
                q_uni_counts[tag] +=1
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

def S(k, q_uni_counts):
        taglist = set()
        for tag in q_uni_counts:
            taglist.add(tag)
        if k in (-1, 0):
            return {"*"}
        else:
            return taglist
def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """

    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    for key in q_tri_counts:
        q_tri_counts[key] =  q_tri_counts[key]*lambda1 + q_bi_counts[(key[0],key[1])]*lambda2 +  q_uni_counts[key[0]]*(1-lambda1 - lambda2)

    START_SYMBOL = "*"
    STOP_SYMBOL = "."
    pi = defaultdict(float)
    bp = {}
    LOG_PROB_OF_ZERO = -1000
    # Initialization
    pi[(0, START_SYMBOL, START_SYMBOL)] = 1.0
    n  = len(sent)
    for k in range(1, n+1):
        for u in S(k-1,q_uni_counts):
            for v in S(k,q_uni_counts):
                max_score = float('-Inf')
                max_tag = None
                for w in S(k - 2, q_uni_counts):
                    if e_word_tag_counts.get((sent[k-1][0], v), 0) != 0:
                        score = pi.get((k-1, w, u), LOG_PROB_OF_ZERO)+q_tri_counts.get((w, u, v), LOG_PROB_OF_ZERO)+e_word_tag_counts.get((sent[k-1][0], v))
                        if score > max_score:
                            max_score = score
                            max_tag = w
                pi[(k, u, v)] = max_score
                bp[(k, u, v)] = max_tag
    max_score = float('-Inf')
    u_max, v_max = None, None
    for u in S(n-1, q_uni_counts):
        for v in S(n, q_uni_counts):
            score = pi.get((n, u, v), LOG_PROB_OF_ZERO) + q_tri_counts.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if score > max_score:
                max_score = score
                u_max = u
                v_max = v

    tags = deque()
    tags.append(v_max)
    tags.append(u_max)

    for i, k in enumerate(range(n-2, 0, -1)):
        tags.append(bp[(k+2, tags[i+1], tags[i])])
    tags.reverse()

    for j in range(0, n):
        predicted_tags[j]  = tags[j]

    ### END YOUR CODE
    print predicted_tags, sent, len(predicted_tags), len(sent)
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    print
    """
    
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    lambda1_best = 0
    lambda2_best = 0

    lambda1 = 0
    step = 0.05
    while lambda1 <= 1.0:
        lambda2 = 0
        while lambda2 <= (1.0 - lambda1):
            acc = 0
            total_length = 0
            for sent in test_data:
                result = hmm_viterbi(sent,total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts,lambda1,lambda2)
                total_length += len(result)
                for i in range(len(result)):
                    if result[i]== sent[i][1]:
                        acc += 1
            acc  = acc*1.0/ total_length
            if acc > acc_viterbi:
                acc_viterbi = acc
                lambda1_best = lambda1
                lambda2_best = lambda2
            lambda2 += step
        lambda1 += step

    print "best parameters lambda1:", lambda1_best, " lambda2:",lambda2_best, " lambda3", 1-(lambda2_best+lambda1_best)
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
    hmm_viterbi(dev_sents[0],total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts,0.5,0.3)
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