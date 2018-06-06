from PCFG import PCFG
import sys

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file(sys.argv[1])
    cnf = pcfg.to_cnf()
    num_of_sents = 1
    print_tree = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "-n":
            num_of_sents = int(sys.argv[3])
        elif sys.argv[2] == "-p":
            cnf_file = sys.argv[3]
            print_tree = True
            cnf.print_grammar(cnf_file)
        else:
            assert 0
    if len(sys.argv) > 4:
        assert sys.argv[4] == "-t"
        for i in xrange(num_of_sents):
            print pcfg.random_tree()
    else:
        if not print_tree: 
            for i in xrange(num_of_sents):
                print pcfg.random_sent()
    