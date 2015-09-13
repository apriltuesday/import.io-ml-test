# April Shen -- 2015-09-10
# The Punctuation Corrector (https://www.hackerrank.com/challenges/punctuation-corrector-its)
import re
from queue import PriorityQueue as PQ


########################## FREQUENCIES ########################


def getNgrams(corpus, frequencies, sub, head, tail):
    """
    Get ngrams of length 2 <= n <= 5 containing a particular
    substitute word (it's or its) and find the number of their
    occurences in the corpus. See Guiliano et al. 2007 on 
    syntagmatic coherence.

    corpus: corpus of text to search for ngrams
    frequencies: priority queue, to rank substitutes by length
    	of ngram and frequency in the corpus
    sub: the candidate substitute word
    head, tail: lists of words preceding and following sub
    """
    ngrams = []
    # hLength and tLength are the maximum number of words we want
    # to include from head and tail, respectively  
    hLength = min(len(head), 4)
    tLength = min(len(tail), 4)
         
    # generate ngrams of lengths 2 <= n <= 5
    for i in range(0, hLength+1):
        for j in range(0, min(4-i+1, tLength+1)):
            ngram = list(head[len(head)-i:]) + [sub] + list(tail[:j])
            ngrams.append(ngram)
    # remove ngram that is just sub alone
    ngrams.remove([sub])

    # for each ngram, find frequency
    for ngram in ngrams:
        freq = searchCorpus(corpus, ngram)
        # if frequency is not zero, add the substitute to the PQ.
        # priority is ranked by 1/length, then 1/frequency
        if freq != 0:
            n = len(ngram)
            frequencies.put(((1.0/n, 1.0/freq), sub))
       

def searchCorpus(corpus, ngram):
	"""
	Search corpus for an ngram and return the number of instances.
	"""
	words = " ".join(ngram)
	return corpus.count(words)


####################### FILE PARSING #########################


def readCorpus():
	"""
	Read in corpus file and return as a single string.
	"""
	with open("corpus.txt", encoding="utf-8") as f:
		corpus = "".join(f.readlines()).lower()
	return corpus


def parseInput(filename):
	"""
	Parse input file.
	Return list of contets (heads and tails), divided by ??? which
	are the locations of it's/its.

	If there are multiple instances of ???, we split them up into
	separate contexts. For example,
	A???B???C => (A,B) and (B,C)
	"""
	delimiter = "???"
	contexts = []
	with open(filename) as f:
		for line in f.readlines()[1:]:
			tup = tuple(line.lower().split(delimiter))
			if len(tup) == 2:
				contexts.append(tup)
			else: # handle sentences with multipe ???
				for i in range(len(tup)-1):
					contexts.append((tup[i], tup[i+1]))
	return contexts


def parseOutput(filename):
	"""
	Parse gold standard output file.
	Return a list of all the it's/its in order.
	"""
	answers = []
	with open(filename) as f:
		for line in f.readlines():
			answers += re.findall("[iI]t'?s", line)
	return [a.lower() for a in answers]


########################## MAIN ###########################


def score(W, T):
	"""
	Compute HackerRank score, defined as follows.

	C = # Correct
	W = # Wrong
	T = Total number of test instances
	Score = 100 * (C-W)/T. 
	"""
	return 100.0 * ((T-W) - W) / T


if __name__ == "__main__":
	# setup
	corpus = readCorpus()
	contexts = parseInput("input-and-output/test-input.txt")
	answers = parseOutput("input-and-output/test-output.txt")
	frequencies = PQ()
	guesses = []

	# search corpus for n-grams containing "it's" and "its" inserted
	# between head and tail
	for head,tail in contexts:
		getNgrams(corpus, frequencies, "its", head, tail)
		getNgrams(corpus, frequencies, "it's", head, tail)
		# also check for "it is" and "it has"
		getNgrams(corpus, frequencies, "it is", head, tail)
		getNgrams(corpus, frequencies, "it has", head, tail)

		# get the top choice
		best = frequencies.get()[1]
		if best == "it is" or best == "it has":
			best = "it's"
		guesses.append(best)

	# print output
	# note this doesn't piece together sentences with multiple it's/its
	for w,(head,tail) in zip(guesses, contexts):
		print(head + w + tail)
	print()

	# compute accuracy and score
	errors = sum([1 if x != y else 0 for x,y in zip(answers, guesses)])
	accuracy = 1.0 - errors / len(answers)
	score = score(errors, len(answers))
	print("Accuracy:", accuracy)
	print("HackerRank score:", score)