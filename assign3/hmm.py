from data import *
from memm import build_tag_to_idx_dict
import time
from submitters_details import get_details
from tester import verify_hmm_model
from collections import defaultdict, deque
import numpy as np

LOG_PROB_OF_ZERO = -1000

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    n_sents = 0
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, \
        e_tag_counts = {}, {}, {}, {}, {}

    ### YOUR CODE HERE
    for i in range(len(sents)):
        n_sents += 1
        first = "*";
        second = "*"
        for word,tag in sents[i]:
            total_tokens += 1
            if tag not in q_uni_counts:
                q_uni_counts[tag] = 1
            else:
                q_uni_counts[tag] += 1
            if (second,tag) not in q_bi_counts:
                 q_bi_counts[(second,tag)] = 1
            else:
                 q_bi_counts[(second,tag)] += 1
            if (first,second,tag) not in q_tri_counts:
                q_tri_counts[(first,second,tag)] = 1
            else:
                q_tri_counts[(first,second,tag)] += 1
            if (word,tag) not in e_word_tag_counts:
                 e_word_tag_counts[(word,tag)] = 1
            else:
                 e_word_tag_counts[(word,tag)] += 1
            if tag not in e_tag_counts:
                e_tag_counts[tag] = 1
            else:
                e_tag_counts[tag] += 1
            first = second
            second = tag

    q_bi_counts[("*","*")] = n_sents
    q_uni_counts["*"] = n_sents

    for key in q_tri_counts: # q(yi/(yi-2,yi-1))
        q_tri_counts[key] =  q_tri_counts[key] * 1.0 / q_bi_counts[(key[0],key[1])]
    for key in q_bi_counts: # q(yi/yi-1)
        q_bi_counts[key] = q_bi_counts[key] * 1.0 / q_uni_counts[key[0]]
    for key in q_uni_counts: # c(w) / total tokens
        q_uni_counts[key] =  q_uni_counts[key] * 1.0 / total_tokens
    for key in e_word_tag_counts: # p(xi/yi)
        e_word_tag_counts[key] = e_word_tag_counts[key] * 1.0 / e_tag_counts[key[1]]
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, \
        e_word_tag_counts, e_tag_counts

def get_q(v, u, w, q_tri_counts, q_bi_counts, q_uni_counts, lambda1, \
    lambda2, index_to_tag_dict):
    """
        v is current tag
        u is previous tag
        w is previous previous tag
    """

    tag_v = index_to_tag_dict[v]
    tag_u = index_to_tag_dict[u]
    tag_w = index_to_tag_dict[w]

    prob = 0.0
    tri = (tag_w, tag_u, tag_v)

    if q_tri_counts.get(tri) != None:
        prob = q_tri_counts[tri] * lambda1

    bi = (tag_u, tag_v)
    if q_bi_counts.get(bi) != None:
        prob+= q_bi_counts[bi] * lambda2

    if q_uni_counts.get(tag_v) != None:
        prob += q_uni_counts[tag_v] * (1 - lambda1 - lambda2)

    if prob <= 0.0:
        prob = LOG_PROB_OF_ZERO 
    return prob

def get_e(word, tag, e_word_tag_counts, index_to_tag_dict):
    prob = 0.0
    pair = (word, index_to_tag_dict[tag])

    if e_word_tag_counts.get(pair) != None:
        prob = e_word_tag_counts[pair]
    else:
        prob = LOG_PROB_OF_ZERO

    # print str.format("word={} tag={} e_prob={}", pair[0], pair[1], prob)
    return prob

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
    e_word_tag_counts, e_tag_counts, lambda1, lambda2, tag_to_index_dict):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """

    # print sent

    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE

    index_to_tag_dict = invert_dict(tag_to_index_dict)

    START_SYMBOL = "*"
    STOP_SYMBOL = "STOP"

    pi = defaultdict(float)
    bp = {}
    
    # Initialization
    pi[(0, tag_to_index_dict[START_SYMBOL], tag_to_index_dict[START_SYMBOL])] = 0.0
    n  = len(sent)
    for k in range(1, n + 1):
        for v in index_to_tag_dict.keys():
            if index_to_tag_dict[v] == START_SYMBOL:
                continue
            # print str.format("word={} v={}", sent[k - 1][0], index_to_tag_dict[v])
            if get_e(sent[k - 1][0], v, e_word_tag_counts, index_to_tag_dict) <=  LOG_PROB_OF_ZERO: # prunning
                continue
            for u in index_to_tag_dict.keys():
                max_score = float('-Inf')
                max_w = None 
                for w in index_to_tag_dict.keys(): 
                    score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO)
                    if score <= LOG_PROB_OF_ZERO:
                        continue 
                    score += get_q(v, u, w, q_tri_counts, q_bi_counts, q_uni_counts, \
                        lambda1, lambda2, index_to_tag_dict) + \
                        get_e(sent[k - 1][0], v, e_word_tag_counts, index_to_tag_dict)

                    # print str.format("word={} v={} u={} w={}", sent[k - 1][0], \
                            # index_to_tag_dict[v], index_to_tag_dict[u], index_to_tag_dict[w])

                    if score > max_score:
                        max_score = score
                        max_w = w

                if max_w != None:
                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_w
                    # print str.format("k={} word={} v={} u={} max_w={} score={}", k, sent[k - 1][0], \
                            # index_to_tag_dict[v], index_to_tag_dict[u], \
                            # index_to_tag_dict[max_w], max_score)

    max_score = float('-Inf')
    u_max, v_max = None, None
    for u in index_to_tag_dict.keys():
        for v in index_to_tag_dict.keys():
            score = pi.get((n, u, v), LOG_PROB_OF_ZERO)          
                # + get_q(STOP_SYMBOL, v, u, q_tri_counts, q_bi_counts, 
                # q_uni_counts, lambda1, lambda2)
            if score > max_score: 
                max_score = score
                u_max = u
                v_max = v

    if len(sent) > 1:
        predicted_tags[-2] = index_to_tag_dict[u_max]
    predicted_tags[-1] = index_to_tag_dict[v_max]

    u = u_max
    v = v_max
    i = 3
    for k in range(n, 2, -1):
        # print str.format("k={} u={} v={}", k, index_to_tag_dict[u], index_to_tag_dict[v])
        w = bp[(k, u, v)]
        predicted_tags[-i] = index_to_tag_dict[w]
        v = u
        u = w
        i += 1

    ### END YOUR CODE
    # print [(sent[i][0], predicted_tags[i]) for i in range(len(sent))]
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
        e_word_tag_counts,e_tag_counts, tag_to_index_dict):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    
    print "Start evaluation"

    acc_viterbi = 0.0
    ### YOUR CODE HERE
    lambda1_best = 0
    lambda2_best = 0

    lambda1 = 0
    step = 0.1

    while lambda1 <= 1.0:
        lambda2 = 0
        while lambda2 <= (1.0 - lambda1):
            acc = 0.0
            total = 0
            match = 0
            start = time.time()
            for sent in test_data:
                pred_tags = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, 
                    q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1, lambda2,
                    tag_to_index_dict)

                total += len(pred_tags)

                for i in range(len(pred_tags)):
                    if pred_tags[i] == sent[i][1]:
                        match += 1
            acc  = match * 1.0 / total
            end = time.time()
            print str.format("lambda1={} lambda2={} score={} time in Sec={}", lambda1, lambda2, acc, end - start)

            if acc > acc_viterbi:
                acc_viterbi = acc
                lambda1_best = lambda1
                lambda2_best = lambda2
                print str.format("best score lambda1={} lambda2={} score={}", lambda1, lambda2, acc)

            lambda2 += step
        lambda1 += step

    print "best parameters lambda1:", lambda1_best, " lambda2:",lambda2_best, \
        " lambda3", 1-(lambda2_best+lambda1_best)
    ### END YOUR CODE

    return str(acc_viterbi)

if __name__ == "__main__":
    print (get_details())
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, \
        e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    """
    pred_tags = hmm_viterbi(dev_sents[0], total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
        e_word_tag_counts, e_tag_counts, 0.5, 0.3, tag_to_idx_dict)

    match = sum([pred_tags[i] == dev_sents[0][i][1] for i in range(len(dev_sents[0]))])

    print "Dev: Accuracy of Viterbi hmm: " + str(float(match) / len(dev_sents[0]))

    verify_hmm_model(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
        e_word_tag_counts, e_tag_counts)
    """

    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, \
        q_uni_counts, e_word_tag_counts, e_tag_counts, tag_to_idx_dict)
    

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