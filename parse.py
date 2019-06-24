#!/usr/bin/env python3
import math
import sys
from collections import namedtuple, defaultdict

Rule = namedtuple('Rule', ['lhs', 'rhs', 'weight'])
Entry = namedtuple('Entry', ['dot_pos', 'start_pos', 'rule', 'wt', 'b_ptrs', 'row_i'])

class EarleyParser:
    """A class implementing the Earley parser.

    Attributes:
        nonterm_rules        (dict): A dict that maps a nonterminal (str) to list of
                                     (nonterminal) rules that have it as the lhs.
        term_rules           (dict): A dict that maps a terminal (str) to list of
                                     (terminal) rules that have it as the rhs.
        nonterm_left_parents (dict): A dict that maps a nonterminal (str) to set of
                                     its "left parents".
    """

    def __init__(self):
        """Initializes class attributes."""
        self.nonterm_rules = defaultdict(list)
        self.term_rules = defaultdict(list)
        self.nonterm_left_parents = defaultdict(set)

    def load_grammar(self, path):
        """Loads grammar file and set up grammar rules.

        Args:
            path (str): Path to grammar file.
        """
        with open(path, 'r') as f:
            for line in f.readlines():
                weight, lhs, rhs = line.split('\t')
                weight = -math.log(float(weight), 2)
                self.nonterm_rules[lhs].append(Rule(lhs, tuple(rhs.split()), weight))

        nonterminals = set(self.nonterm_rules.keys())

        for lhs in nonterminals:
            rule_i = 0
            while rule_i < len(self.nonterm_rules[lhs]):
                rhs_head = self.nonterm_rules[lhs][rule_i].rhs[0]
                if rhs_head in nonterminals:
                    self.nonterm_left_parents[rhs_head].add(lhs)
                    rule_i += 1
                else:
                    term_rule = self.nonterm_rules[lhs].pop(rule_i)
                    self.term_rules[rhs_head].append(term_rule)
            if len(self.nonterm_rules[lhs]) == 0:
                del self.nonterm_rules[lhs]


    def parse_file(self, path):
        """Parses sentences in a given file and prints resulting parses.

        Args:
            path (str): Path to file to parse.
        """
        with open(path, 'r') as f:
            for sentence in f.readlines():
                result = self.parse_sentence(sentence)
                print(result)

    def parse_sentence(self, sentence):
        """Parses a given sentence.

        Args:
            sentence (str): A sentence to parse.

        Returns:
            str: The resulting parse and weight; "NONE" if no valid parse.
        """
        sent_parser = self.SentenceParser(sentence, \
                                          self.nonterm_rules, \
                                          self.term_rules, \
                                          self.nonterm_left_parents)
        return sent_parser.parse()

    class SentenceParser:
        """A class for parsing a sentence using per-sentence configurations.

        Attributes:
            sentence     (list): The sentence to parse (as a list).
            s_len         (int): Length of sentence.
            rules        (dict): A dict that maps a nonterminal (str) to list of
                                 rules that have it as the lhs.
            left_parents (dict): A dict that maps first child of a rule (str) to
                                 set of its "left parents".
            parse_t      (list): The parsing table as a nested list;
                                 `parse_t[i][j]` represents entry in col i row j.
        """

        def __init__(self, sent, nonterm_rules, term_rules, nonterm_left_parents):
            """Initializes class attributes.

            Args:
                sent                  (str): The sentence to parse.
                nonterm_rules        (dict): A dict that maps a nonterminal (str) to list of
                                             (nonterminal) rules that have it as the lhs.
                term_rules           (dict): A dict that maps a terminal (str) to list of
                                             (terminal) rules that have it as the rhs.
                nonterm_left_parents (dict): A dict that maps a nonterminal (str) to set of
                                             its "left parents".
            """
            self.sentence = sent.strip('\n').split()
            self.s_len = len(self.sentence)

            # Init `rules` and `left_parents` w/ only nonterminal entries
            self.rules = nonterm_rules.copy()
            self.left_parents = nonterm_left_parents.copy()

            # Update `rules` and `left_parents` so that they also contain
            # terminal entries relating to words in the current sentence. I.e.,
            # 1) terminal rules containing some word in the sentence
            # 2) left parents of some word in the sentence
            for word in set(self.sentence):
                for rule in term_rules.get(word, []):
                    self.rules[rule.lhs].append(rule)
                    self.left_parents[word].add(rule.lhs)

            # Init parse table `parse_t` w/ entries sprawned from 'ROOT'
            self.parse_t = [[]]
            for root_rule in self.rules['ROOT']:
                new_ent = Entry(0, 0, root_rule, root_rule.weight, \
                                [None]*len(root_rule.rhs), len(self.parse_t[0]))
                self.parse_t[0].append(new_ent)

        def parse(self):
            """Parses the given sentence.

            Returns:
                str: The resulting parse and weight; "NONE" if no valid parse.
            """
            # No valid parse if empty sentence
            if self.s_len == 0:
                return 'NONE'

            # `parse_t[col_i]` contains entries that have i words processed
            col_i = 0
            while (col_i < len(self.parse_t)) and (col_i <= self.s_len):
                row_i = 0
                dup_dict = {}
                pred_set = set()
                left_ancestors = set()

                # Set up left ancestors lookup
                if col_i < self.s_len:
                    self._find_left_ancestors(self.sentence[col_i], left_ancestors)

                while row_i < len(self.parse_t[col_i]):
                    if self.parse_t[col_i][row_i] is None:
                        row_i += 1
                        continue
                    ent = self.parse_t[col_i][row_i]
                    if ent.dot_pos == len(ent.rule.rhs):
                        # Attach if rule is completed
                        self._attach(col_i, row_i, dup_dict)
                    elif ent.rule.rhs[ent.dot_pos] not in self.rules:
                        # Scan if a terminal is encountered
                        self._scan(col_i, row_i)
                    else:
                        # Predict if a nonterminal is encountered
                        self._predict(col_i, row_i, dup_dict, pred_set, left_ancestors)
                    row_i += 1
                col_i += 1

            # No valid parse if didn't cover the whole sentence
            if len(self.parse_t) != (len(self.sentence) + 1):
                return 'NONE'

            # Find the lightest completed entry of a ROOT rule
            final_parse = None
            for cand in self.parse_t[-1]:
                if cand is None:
                    # Skip if a null entry
                    continue
                if (cand.dot_pos != len(cand.rule.rhs) or
                    cand.start_pos != 0 or
                    cand.rule.lhs != 'ROOT'):
                    # Skip if not a completed entry of a ROOT rule
                    continue
                if (final_parse is not None and
                    cand.wt >= final_parse.wt):
                   # Skip if not the lightest
                   continue
                final_parse = cand

            # No valid parse for the sentence
            if final_parse is None:
                return 'NONE'

            output = []
            self._gen_output(zip(final_parse.rule.rhs, final_parse.b_ptrs), output)
            return '(ROOT ' + ' '.join(output) + ')\n' + str(final_parse.wt)

        def _attach(self, col_i, row_i, dup_dict):
            """Attaches the current completed entry to a previous entry if valid.

            E.g., Suppose current entry contains rule 'NP -> Det N.' w/ start_pos = 3,
                  then if col 3 contains an entry where the next term (term
                  after the dot) is 'NP', such as 'S -> .NP VP', attach succeeds
                  and a new entry 'S -> NP .VP' added to the current column.

            Args:
                col_i          (int): Column index of the current entry.
                row_i          (int): Row index of the current entry.
                dup_dict      (dict): A dict that maps a tuple of an entry's
                                      (dot_pos, start_pos, rule) to the entry.
            """
            ent = self.parse_t[col_i][row_i]
            for cand in self.parse_t[ent.start_pos]:
                if cand is None:
                    # Skip candidate if set to None b/c there was
                    # some other lighter duplicate entry
                    continue
                if cand.dot_pos >= len(cand.rule.rhs):
                    # Skip if candidate was completed
                    continue
                if ent.rule.lhs == cand.rule.rhs[cand.dot_pos]:
                    # Attach if the current completed entry completes
                    # what's after the dot in the candidate's rule
                    new_back_ptrs = cand.b_ptrs.copy()
                    new_back_ptrs[cand.dot_pos] = (col_i, row_i)

                    new_ent = Entry(cand.dot_pos+1, cand.start_pos, \
                                    cand.rule, cand.wt+ent.wt, \
                                    new_back_ptrs, len(self.parse_t[col_i]))

                    # Only attach lighter ones
                    dup_ent = dup_dict.get(new_ent[:3], None)
                    if dup_ent is not None:
                        if new_ent.wt >= dup_ent.wt:
                            continue
                        # Set the dup entry to None since the new entry is lighter
                        self.parse_t[col_i][dup_ent.row_i] = None

                    self.parse_t[col_i].append(new_ent)
                    dup_dict[new_ent[:3]] = new_ent

        def _scan(self, col_i, row_i):
            """Scans the next term in the current entry, which is a terminal, and
               completes the rule if the terminal matches the next word.

            E.g., Suppose the next word to process in the sentence is 'the', then
                  if current entry contains rule 'Det -> .the', the scan succeeds
                  and a new entry containing rule 'Det -> the.' added to new column.
                  (If current entry contains rule 'Det -> .a' instead, the scan fails.)

            Args:
                col_i     (int): Column index of the current entry.
                row_i     (int): Row index of the current entry.
            """
            ent = self.parse_t[col_i][row_i]
            # No scanning if all words are processed
            if col_i >= self.s_len:
                row_i += 1
                return
            # Complete the rule if it's a match
            if (ent.rule.rhs[ent.dot_pos] == self.sentence[col_i]):
                # Create new column if this is the last column
                if (col_i + 1) == len(self.parse_t):
                    self.parse_t.append([])
                # Append to next column
                new_ent = Entry(ent.dot_pos+1, ent.start_pos, \
                                ent.rule, ent.wt, ent.b_ptrs, \
                                len(self.parse_t[col_i+1]))
                self.parse_t[col_i+1].append(new_ent)

        def _predict(self, col_i, row_i, dup_dict, pred_set, left_ancestors):
            """Predicts how the next term in current entry, which is a nonterminal,
               may be expanded.

            E.g., Suppose current entry contains rule 'S -> .NP VP', then
                  the next nonterminal (the one after the dot) is 'NP', and
                  possible expansions can be 'NP -> .Det N', 'NP -> .NP PP', etc.

            Args:
                col_i          (int): Column index of the current entry.
                row_i          (int): Row index of the current entry.
                dup_dict      (dict): A dict that maps a tuple of an entry's
                                      (dot_pos, start_pos, rule) to the entry.
                pred_set       (set): A set of nonterminals that have already been
                                      expanded in current column.
                left_ancestors (set): A set containing left ancestors of next word.
            """
            ent = self.parse_t[col_i][row_i]
            nt_to_predict = ent.rule.rhs[ent.dot_pos]
            # Check whether predicting same stuff
            if nt_to_predict in pred_set:
                row_i += 1
                return

            for cand_rule in self.rules[nt_to_predict]:
                # Check for duplicates
                if (0, col_i, cand_rule) in dup_dict:
                    continue
                if col_i < self.s_len:
                    if cand_rule.lhs not in left_ancestors:
                        continue
                new_ent = Entry(0, col_i, cand_rule, cand_rule.weight, \
                                [None]*len(cand_rule.rhs), \
                                len(self.parse_t[col_i]))
                self.parse_t[col_i].append(new_ent)
                dup_dict[(0, col_i, cand_rule)] = new_ent
                pred_set.add(nt_to_predict)

        def _find_left_ancestors(self, child, left_ancestors):
            """Finds the left ancestors of the first child of a rule.

            Args:
                child          (str): The first child of a rule.
                left_ancestors (set): A set containing left ancestors of next word.
            """
            if child in self.left_parents:
                for parent in self.left_parents[child]:
                    if parent in left_ancestors:
                        continue
                    left_ancestors.add(parent)
                    self._find_left_ancestors(parent, left_ancestors)

        def _gen_output(self, back_ptrs, output):
            """Generates the output parse string by following back pointers.

            Args:
                back_ptrs (list): The back pointers of the current entry.
                output    (list): The output parse string (as a list).
            """
            for term_i, back_ptr in enumerate(back_ptrs):
                term, pos = back_ptr
                if pos is None:
                    output.append(term)
                else:
                    ent = self.parse_t[pos[0]][pos[1]]
                    output.extend(['(', ent.rule.lhs])
                    self._gen_output(zip(ent.rule.rhs, ent.b_ptrs), output)
                    output.append(')')


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 parse.py grammar_file [file_to_parse]')
        sys.exit(1)

    parser = EarleyParser()
    parser.load_grammar(sys.argv[1])

    if len(sys.argv) > 2:
        parser.parse_file(sys.argv[2])
    else:
        while True:
            try:
                usr_in = input('Enter a sentence to parse (Ctrl+C to exit): ')
            except KeyboardInterrupt:
                print()
                sys.exit(1)
            else:
                result = parser.parse_sentence(usr_in)
                print(result)


if __name__ == '__main__':
    main()
