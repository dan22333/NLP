from data import *
from memm import build_tag_to_idx_dict
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

    n_sents = 0
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, \
        e_tag_counts = {}, {}, {}, {}, {}

    ### YOUR CODE HERE
    for i in range(len(sents)):
        n_sents += 1
        first = "*"
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

        key = (first, second, 'STOP')
        if key in q_tri_counts:
            q_tri_counts[key] += 1
        else:
            q_tri_counts[key] = 1

    q_bi_counts[("*","*")] = n_sents
    q_uni_counts["*"] = n_sents

    for key in q_tri_counts: # q(yi/(yi-2,yi-1))
        q_tri_counts[key] =  np.log([q_tri_counts[key] * 1.0 / q_bi_counts[(key[0],key[1])]])[0]
    for key in q_bi_counts: # q(yi/yi-1)
        q_bi_counts[key] = np.log([q_bi_counts[key] * 1.0 / q_uni_counts[key[0]]])[0]
    for key in q_uni_counts: # c(w) / total tokens
        q_uni_counts[key] =  np.log([q_uni_counts[key] * 1.0 / total_tokens])[0]
    for key in e_word_tag_counts: # p(xi/yi)
        e_word_tag_counts[key] = np.log([e_word_tag_counts[key] * 1.0 / e_tag_counts[key[1]]])[0]
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, \
        e_word_tag_counts, e_tag_counts

# Tags are defined by their names
# Used for (STOP, v, u) triple
def get_tri_q(tag_v, tag_u, tag_w, q_tri_counts, lambda1):
    prob = -np.inf
    tri = (tag_w, tag_u, tag_v)

    if q_tri_counts.get(tri) != None:
        prob = q_tri_counts[tri] * lambda1

    return prob

def get_q(v, u, w, q_tri_counts, q_bi_counts, q_uni_counts, lambda1, \
    lambda2, index_to_tag_dict):
    """
        v is current tag index
        u is previous tag index
        w is previous previous tag index
    """

    tag_v = index_to_tag_dict[v]
    tag_u = index_to_tag_dict[u]
    tag_w = index_to_tag_dict[w]

    prob = -np.inf
    tri = (tag_w, tag_u, tag_v)

    if q_tri_counts.get(tri) != None:
        prob = q_tri_counts[tri] * lambda1

    bi = (tag_u, tag_v)
    if q_bi_counts.get(bi) != None:
        prob+= q_bi_counts[bi] * lambda2

    if q_uni_counts.get(tag_v) != None:
        prob += q_uni_counts[tag_v] * (1 - lambda1 - lambda2)

    return prob

def get_e(word, tag, e_word_tag_counts, index_to_tag_dict):
    prob = -np.inf
    pair = (word, index_to_tag_dict[tag])

    if e_word_tag_counts.get(pair) != None:
        prob = e_word_tag_counts[pair]

    return prob

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
    e_word_tag_counts, e_tag_counts, lambda1, lambda2, tag_to_index_dict):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """

    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE

    index_to_tag_dict = invert_dict(tag_to_index_dict)

    START_SYMBOL = "*"
    STOP_SYMBOL = "STOP"

    pi = defaultdict(float)
    bp = {}

    # Initialization
    pi[(0, tag_to_index_dict[START_SYMBOL], tag_to_index_dict[START_SYMBOL])] = 0.0
    n = len(sent)

    # Store the last k. k may not necessarily reach n if at some k0 < n
    # pi[k0,u,v] for all u,v is zero 
    last_k = 0 

    for k in range(1, n + 1):
        for v in index_to_tag_dict.keys():
            if index_to_tag_dict[v] == START_SYMBOL:
                continue
            e_val = get_e(sent[k - 1][0], v, e_word_tag_counts, index_to_tag_dict)
            if e_val <= -np.inf: # prunning
                continue
            for u in index_to_tag_dict.keys():
                max_score = -np.inf
                max_w = None 
                for w in index_to_tag_dict.keys(): 
                    prev_val = pi.get((k - 1, w, u), -np.inf)
                    if prev_val <= -np.inf:
                        continue 
                    score = prev_val + get_q(v, u, w, q_tri_counts, q_bi_counts, 
                        q_uni_counts, lambda1, lambda2, index_to_tag_dict) + \
                        get_e(sent[k - 1][0], v, e_word_tag_counts, index_to_tag_dict)

                    if score > max_score:
                        max_score = score
                        max_w = w

                if max_w != None:
                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_w
                    last_k = k

        # pi[k,_,_] was not built
        if k > last_k:
            break;

    max_score = -np.inf
    u_max, v_max = None, None
    for u in index_to_tag_dict.keys():
        for v in index_to_tag_dict.keys():
            score = pi.get((last_k, u, v), -np.inf)
            if score <= -np.inf:
                continue
            if last_k == n:
                score += get_tri_q(STOP_SYMBOL, index_to_tag_dict[v], 
                    index_to_tag_dict[u], q_tri_counts, lambda1)    
            if score <= -np.inf:
                continue
            if score > max_score: 
                max_score = score
                u_max = u
                v_max = v

    k = last_k

    if k > 1:
        predicted_tags[k - 2] = index_to_tag_dict[u_max]
    predicted_tags[k - 1] = index_to_tag_dict[v_max]

    u = u_max
    v = v_max

    while k > 2:
        w = bp[(k, u, v)]
        predicted_tags[k - 3] = index_to_tag_dict[w]
        v = u
        u = w
        k -= 1

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
        e_word_tag_counts,e_tag_counts, lambda1, lambda2, tag_to_index_dict):

    total = 0
    match = 0
    count = 0

    for sent in test_data:
        pred_tags = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, 
            q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1, lambda2,
            tag_to_index_dict)

        total += len(pred_tags)
        match += sum([pred_tags[i] == sent[i][1] for i in range(len(sent))])

    return match * 1.0 / total

def grid_search(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, 
        e_word_tag_counts,e_tag_counts, tag_to_index_dict):
    """
        Receives: test data set and the parameters learned by hmm
        an evaluation of the accuracy of hmm
    """

    acc_viterbi = 0.0
    ### YOUR CODE HERE
    lambda1_best = 0
    lambda2_best = 0

    lambda1 = 0
    step = 0.1

    while lambda1 <= 1.0:
        lambda2 = 0
        while lambda2 <= (1.0 - lambda1):
            start = time.time()
            acc = hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, 
                q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1, lambda2,
                tag_to_index_dict)

            end = time.time()
            print str.format("lambda1={} lambda2={} score={} time in Sec={}", 
                lambda1, lambda2, acc, end - start)

            if acc > acc_viterbi:
                acc_viterbi = acc
                lambda1_best = lambda1
                lambda2_best = lambda2

            lambda2 += step
        lambda1 += step

    print str.format("best parameters lambda1={} lambda2={} lambda3={}:", 
        lambda1_best, lambda2_best, 1.0 - lambda1_best - lambda2_best)

    ### END YOUR CODE

    return lambda1_best, lambda2_best

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

    lambda1, lambda2 = grid_search(dev_sents, total_tokens, q_tri_counts, q_bi_counts,
        q_uni_counts, e_word_tag_counts, e_tag_counts, tag_to_idx_dict)

    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts,
        q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1, lambda2, tag_to_idx_dict)
 
    print "Dev: Accuracy of Viterbi hmm: " + str(acc_viterbi)

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