from collections import defaultdict
import random
import sys

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    def to_cnf(self):
        grammar1 = PCFG()
        cnf = PCFG()
         # add a new rule X_a -> a to each terminal a
        for lhs, rhs_and_weights in self._rules.iteritems():
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1: # A -> terminal 
                    grammar1._rules[lhs].append((rhs, weight))
                    grammar1._sums[lhs] += weight
                    continue
                for i in range(len(rhs)):
                    if self.is_terminal(rhs[i]):
                        a = rhs[i]
                        X = "X" + "_" + a
                        grammar1._rules[X] = [([a], 1.0)]
                        grammar1._sums[X] = [([a], 1.0)]

        # to each rule A -> X1X2.. where Xi can be either non terminal/terminal variable 
        # replace the occurrences of terminal a with X_a
        for lhs, rhs_and_weights in self._rules.iteritems():
            rhs_and_weights1 = []
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1: # A -> terminal
                    rhs_and_weights1.append((rhs, weight))
                    continue
                var_list = []
                for i in range(len(rhs)):
                    if self.is_terminal(rhs[i]):
                        new_var = "X" + "_" + rhs[i]
                    else:
                        new_var = rhs[i]
                    var_list.append(new_var)

                rhs_and_weights1.append((var_list, weight))

            grammar1._rules[lhs] = rhs_and_weights1
            grammar1._sums[lhs] = self._sums[lhs]

        for lhs, rhs_and_weights in grammar1._rules.iteritems():
             for rhs, weight in rhs_and_weights:
                if len(rhs) <= 2:
                    cnf._rules[lhs].append((rhs, weight))
                    cnf._sums[lhs] += weight
                else:
                    n = len(rhs)
                    lhs1 = lhs
                    weight1 = weight
                    for i in range(0, n - 2):
                        X = rhs[i]
                        Y = self.cnf_create_variable(rhs, i + 1)
                        cnf._rules[lhs1].append(([X, Y], weight))
                        # print(str(lhs1) + " " + str(cnf._rules[lhs1]))
                        cnf._sums[lhs1] += weight
                        weight1 = 1.0
                        lhs1 = Y
                    cnf._rules[lhs1].append(([rhs[n - 2], rhs[n - 1]], weight))
                    cnf._sums[lhs1] += weight
                    # print(str(lhs1) + " " + str(cnf._rules[lhs1]))
        return cnf

    def cnf_create_variable(cls, rhs, i):
        """
            For example if rhs = NP PP VP PP and i = 1
            Y = NP|PP.VP.PP
        """   
        Y = rhs[0]
        for j in range(1, len(rhs)):
            Y += "|" if j == i else "."
            Y += rhs[j]

        return Y

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def print_grammar(self, filename):
        with open(filename, "w") as fh:
            for lhs, rhs_and_weights in self._rules.iteritems():
                for rhs, weight in rhs_and_weights:
                    fh.write(str(weight) + " " + lhs + " " + " ".join(rhs) + "\n")
            fh.flush()

    @classmethod
    def from_file_assert_cnf(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                if len(r) > 2:
                    raise Exception("Grammar is not CNF, right-hand-side is: " + str(r))
                if len(r) <= 0:
                    raise Exception("Grammar is not CNF, right-hand-side is empty: " + str(r))
                grammar.add_rule(l,r,w)
        for lhs, rhs_and_weights in grammar._rules.iteritems():
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1 and not grammar.is_terminal(rhs[0]):
                    raise Exception("Grammar has unary rule: " + str(rhs))
                elif len(rhs) == 2 and (grammar.is_terminal(rhs[0]) or grammar.is_terminal(rhs[1])):
                    raise Exception("Grammar has binary rule with terminals: " + str(rhs))

        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def is_preterminal(self, rhs):
        return len(rhs) == 1 and self.is_terminal(rhs[0])

    def gen(self, symbol):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            return " ".join(self.gen(s) for s in expansion)

    def gentree(self, symbol):
        """
            Generates a derivation tree from a given symbol
        """        
        ### YOUR CODE HERE
        tree = "(" + symbol + " "
        expansion = self.random_expansion(symbol)
        for s in expansion:
            if self.is_terminal(s):
                tree += " " + s
            else:
                tree += " " + self.gentree(s)
        tree += ")"
        ### END YOUR CODE
        return tree

    def random_sent(self):
        return self.gen("ROOT")

    def random_tree(self):
        return self.gentree("ROOT")

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

