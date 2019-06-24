import unittest
import os
from parse import EarleyParser

class TestEarleyParser(unittest.TestCase):
    def setUp(self):
        self.parser = EarleyParser()
        grammar =  "1	ROOT	S\n\
                    1	S	NP VP\n\
                    0.8	NP	Det N\n\
                    0.1	NP	NP PP\n\
                    0.7	VP	V NP\n\
                    0.3	VP	VP PP\n\
                    1	PP	P NP\n\
                    0.1	NP	Papa\n\
                    0.5	N	caviar\n\
                    0.5	N	spoon\n\
                    1	V	ate\n\
                    1	P	with\n\
                    0.5	Det	the\n\
                    0.5	Det	a"
        with open('test.gr', 'w') as f:
            f.write(grammar)
        self.parser.load_grammar('test.gr')
        self.maxDiff = None

    def test_sent_valid(self):
        """Tests that a sentence with a valid parse is parsed correctly."""
        sentence = 'Papa ate the caviar with a spoon'
        parse_str, weight = self.parser.parse_sentence(sentence).split('\n')
        self.assertEqual(parse_str, '(ROOT ( S ( NP Papa ) ( VP ( VP ( V ate ) ( NP ( Det the ) ( N caviar ) ) ) ( PP ( P with ) ( NP ( Det a ) ( N spoon ) ) ) ) ))')
        self.assertAlmostEqual(float(weight), 10.2173230517)

    def test_sent_invalid(self):
        """Tests that a sentence without a valid parse gets output 'NONE'."""
        sentence = 'Papa the caviar with a spoon'
        result = self.parser.parse_sentence(sentence)
        self.assertEqual(result, 'NONE')

    def tearDown(self):
        os.remove('test.gr')

if __name__ == '__main__':
    unittest.main()
