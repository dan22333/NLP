from PCFG import PCFG
import numpy as np
import math
import copy

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def invert_dict(dict):
    res = {}
    for k, v in dict.items():
        res[v] = k
    return res

def build_non_terminal_to_index_dict(pcfg):
    curr_rule_index = 0
    non_terminal_to_idx_dict = {}
    for rule in pcfg._rules:
        non_terminal_to_idx_dict[rule] = curr_rule_index
        curr_rule_index += 1

    return non_terminal_to_idx_dict

def cky(pcfg, sent, non_terminal_to_idx_dict, idx_to_non_terminal_dict):
    # print(sent)
    words = sent.split(' ')
    ### YOUR CODE HERE
    bp = {} # key : (i, j, X) , value : (rule_ind, rhs_ind, split_point)
    pi = np.zeros((len(words), len(words), len(non_terminal_to_idx_dict)))

    # initialization
    n = len(words)
    m = len(non_terminal_to_idx_dict)
    for i in range(n):
        for X_ind in range(m): 
            X = idx_to_non_terminal_dict[X_ind]
            for rhs, w in pcfg._rules[X]:
                if len(rhs) == 1:
                    assert pcfg.is_terminal(rhs[0])
                    if rhs[0] == words[i]:
                        pi[i, i, X_ind] = w / pcfg._sums[X]

    for l in range(1, n):
        for i in range(0, n - l):
            j = i + l
            for X_ind in range(m):
                X = idx_to_non_terminal_dict[X_ind]
                best_s = -1 # best split-point
                [best_Y, best_Z] = ['None', 'None'] 
                # [(rhs, w),..]
                for rhs_and_w in pcfg._rules[X]:
                    rhs, w = rhs_and_w
                    if pcfg.is_preterminal(rhs):
                            continue
                    [Y, Z] = rhs # X -> YZ
                    Y_ind = non_terminal_to_idx_dict[Y]
                    Z_ind = non_terminal_to_idx_dict[Z]
                    for s in range(i, j): # split point
                        prob = pi[i, s, Y_ind] * pi[s + 1, j, Z_ind] * w / pcfg._sums[X]
                        # print(str(i) + " " + str(j) + " " + str(prob) + " " + X + " " + Y + " " + Z \
                        # + " " + str(s) + " " + str(pi[i, s, Y_ind]) + " " + str(pi[s + 1, j, Z_ind]))
                        if pi[i, j, X_ind] < prob:
                            pi[i, j, X_ind] = prob
                            best_Y, best_Z = Y, Z
                            best_s = s

                if pi[i, j, X_ind] > 0:
                    # print("bp " + str(i) + " " + str(j) + " " + X + " = " + best_Y + " " + best_Z + " " + str(best_s))
                    bp[(i, j, X)] = (best_Y, best_Z, best_s)
    
    key = (0, n - 1, "ROOT")
    if bp.has_key(key):           
        tree = gentree_from_bp(pcfg, bp, words, 0, n - 1, "ROOT")
        if tree != None:
            return tree

    ### END YOUR CODE
    return "FAILED TO PARSE!"

def gentree_from_bp(pcfg, bp, sent, i, j, symbol):
    if i == j: 
        return "(" + symbol + " " + sent[i] + ")"

    key = (i, j, symbol)
    if not bp.has_key(key):
        return None

    (Y, Z, s) = bp[key]
    rhs1 = gentree_from_bp(pcfg, bp, sent, i , s, Y)
    rhs2 = gentree_from_bp(pcfg, bp, sent, s + 1 , j, Z)

    # Failed to parse
    if rhs1 == None or rhs2 == None:
        return None

    return "(" + symbol + " " + rhs1 + " " + rhs2 + ")"

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    non_terminal_to_idx_dict = build_non_terminal_to_index_dict(pcfg)
    idx_to_non_terminal_dict = invert_dict(non_terminal_to_idx_dict)
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent, non_terminal_to_idx_dict, idx_to_non_terminal_dict)
