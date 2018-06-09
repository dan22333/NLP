from collections import defaultdict
import random
import sys

class DG(object): # directed graph
    def __init__(self):
        self._edges = defaultdict(list)
        self._vertices_in_degree = defaultdict(int) # num edges entering to vertex
        self._vertices_out_degree = defaultdict(int) # num edges exiting vertex
        self._zero_in_degree_vertices = {} 
        self._zero_out_degree_vertices = {} 
        self._vertices = {}

    def add_edge(self, u, v): # u->v 
        self._edges[u].append(v)
        self._vertices[u] = True
        self._vertices[v] = True

        if self._zero_out_degree_vertices.has_key(u):
            del self._zero_out_degree_vertices[u]
        if self._zero_in_degree_vertices.has_key(v):
            del self._zero_in_degree_vertices[v]

        self._vertices_in_degree[v] += 1
        self._vertices_out_degree[u] += 1

        if u not in self._vertices_in_degree:
            self._zero_in_degree_vertices[u] = True
        if v not in self._vertices_out_degree:
            self._zero_out_degree_vertices[v] = True

        # print("edge {}->{}".format(u, v))
        # print("in_degree {}".format(self._vertices_in_degree))
        # print("zero in degree {}".format(self._zero_in_degree_vertices))
        # print("zero out degree {}".format(self._zero_out_degree_vertices))
        # print("edges {}".format(self._edges))

    def top_level_vertices(self):
        return self._zero_in_degree_vertices.keys()

    def is_leaf(self, u):
        return self._zero_out_degree_vertices.has_key(u)

    def is_top_level_vertex(self, u):
        return self._zero_in_degree_vertices.has_key(u)

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    # Unsed for replacing unit rules
    def traverse_graph(self, grammar, graph, top_lhs, lhs, weight):
        # print("top_lhs {} lhs {}".format(top_lhs, lhs))
        for rhs, rhs_weight in self._rules[lhs]:
            if len(rhs) == 1 and not self.is_preterminal(rhs):
                mult_weight = weight * rhs_weight 
                if graph.is_leaf(rhs[0]):
                    for rhs1, weight1 in self._rules[rhs[0]]:
                        grammar._rules[top_lhs].append((rhs1, \
                            mult_weight * weight1 / self._sums[rhs[0]]))
                else:
                    assert rhs[0] in graph._edges
                    self.traverse_graph(grammar, graph, top_lhs, rhs[0], mult_weight / self._sums[lhs])

    def to_cnf(self):
        grammar1 = PCFG()
        cnf = PCFG()
        
        # replace unit rules
        unit_rules_graph = DG()
        for lhs, rhs_and_weights in self._rules.iteritems():
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1 and not self.is_preterminal(rhs):
                    unit_rules_graph.add_edge(lhs, rhs[0])

        for lhs in unit_rules_graph.top_level_vertices():
            self.traverse_graph(grammar1, unit_rules_graph, lhs, lhs, 1.0)

        # print("grammar1 rules {}".format(grammar1._rules))

        # add a new rule X_a -> a to each terminal a
        for lhs, rhs_and_weights in self._rules.iteritems():
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1:
                    # if self.is_preterminal(rhs):  # A -> terminal
                    # grammar1._rules[lhs].append((rhs, weight))
                    # grammar1._sums[lhs] += weight
                    continue
                for i in range(len(rhs)):
                    if self.is_terminal(rhs[i]):
                        a = rhs[i]
                        X = "X" + "_" + a
                        grammar1._rules[X] = [([a], 1.0)]
                        grammar1._sums[X] = 1.0

        # print("grammar1 rules {}".format(grammar1._rules))

        # to each rule A -> X1X2.. where Xi can be either non terminal/terminal variable 
        # replace the occurrences of terminal a with X_a
        for lhs, rhs_and_weights in self._rules.iteritems():
            rhs_and_weights1 = []
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1: # A -> terminal | B
                    if self.is_preterminal(rhs):
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

            grammar1._rules[lhs] += rhs_and_weights1
            grammar1._sums[lhs] = self._sums[lhs]

            # update weights due to the removal of unit rules 
            for rhs, weight in rhs_and_weights:
                if len(rhs) == 1 and not self.is_preterminal(rhs): # A -> B
                    if not unit_rules_graph.is_top_level_vertex(lhs):
                        assert grammar1._rules.has_key(lhs)
                        grammar1._sums[lhs] -= weight
                        if grammar1._sums[lhs] <= 0.0:
                            del grammar1._sums[lhs]

        # print("grammar1 rules {}".format(grammar1._rules))
        # print("grammar1 sums {}".format(grammar1._sums))

        # replace X -> X1X2X3.. to X -> X1 X1|X2.X3.. and  
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

