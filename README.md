# An Earley parser implementation.

## Usage
python3 parse.py grammar\_file [file\_to\_parse]

If file\_to\_parse provided, parses each sentence in the file and prints resulting parses and weights.

If file\_to\_parse not provided, enters interactive mode where user's prompted for sentences to be parsed.

## Input format
A grammar\_file contains grammar rules, where each line is a grammar rule containing

1. probability of the rule  
2. left-hand-side of the rule  
3. right-hand-side of the rule

, separated by tabs.

A file\_to\_parse contains sentences, where each line is a sentence, and words are separated by spaces.

## Output format
If there's a valid parse for a given sentence, the output contains two lines: 

1. The resulting parse tree represented as a bracketed string
2. The weight of the parse (negative log probability)

If there's no valid parse for a given sentence, the output would be "NONE".

## Implementation
The script implements the Earley parser w/ left corner filtering. It is implemented based on the algorithm described [here](http://www.cs.jhu.edu/~jason/465/PDFSlides/lect10-earley.pdf).