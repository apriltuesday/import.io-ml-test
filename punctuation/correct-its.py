# April Shen -- 2015-09-11
# The Punctuation Corrector (https://www.hackerrank.com/challenges/punctuation-corrector-its)
import re
from queue import PriorityQueue as PQ


def getNgrams(corpus, frequencies, sub, head, tail):
    """
    get bounds of head and tail (4 or fewer words away from end
    where substitute goes). ngrams will store list of ngrams for
    a particular substitute and context. We will care about ngrams
    with length 2 <= n <= 5
    """
    ngrams = []
  
    # head and tail are stored as lists of words parsed from context
    # sentence (substitute goes between the two)
    # hLength and tLength are the maximum number of words we want
    # to include from head and tail, respectively  
    hLength = min(len(head), 4)
    tLength = min(len(tail), 4)
         
    # Generates ngrams of lengths 2 <= n <= 5
    for i in range(0, hLength+1):
        for j in range(0, min(4-i+1, tLength+1)):
            ngram = list(head[len(head)-i:]) + [sub] + list(tail[:j])
            ngrams.append(ngram)
    #remove ngram that is just sub
    ngrams.remove([sub])

    # for each ngram, find frequency
    for ngram in ngrams:
        freq = searchCorpus(corpus, ngram)
        # if frequency is not zero, add the ngram triple to
        #frequencies PQ. Priority is ranked by 1/n, then 1/frequency
        if freq != 0:
            n = len(ngram)
            frequencies.put(((1.0/n, 1.0/freq), (sub, n, freq)))
       

def searchCorpus(corpus, ngram):
	"""
	Searches corpus for number of instances of an ngram.
	"""
	words = " ".join(ngram)
	count = corpus.count(words)
	return count


def readCorpus():
	"""
	Read in corpus file and return as a single string.
	"""
	with open("corpus.txt", encoding="utf-8") as f:
		corpus = "".join(f.readlines()).lower()
	return corpus


def parseInput(filename):
	"""
	Parses input file.
	Return list of heads and tails, divided by ??? which are the 
	locations of it's/its.
	"""
	delimiter = "???"
	parsed = []
	with open(filename) as f:
		for line in f.readlines()[1:]:
			tup = tuple(line.lower().split(delimiter))
			if len(tup) == 2:
				parsed.append(tup)
			else: # handle sentences with multipe ???
				for i in range(len(tup)-1):
					parsed.append((tup[i], tup[i+1]))
	return parsed


def parseOutput(filename):
	"""
	Parses output file.
	Return a list of all the it's/its in order.
	"""
	answers = []
	with open(filename) as f:
		for line in f.readlines():
			answers += re.findall("it'?s", line)
	return [a.lower() for a in answers]


if __name__ == "__main__":
	# setup
	corpus = readCorpus()
	contexts = parseInput("input-and-output/test-input.txt")
	answers = parseOutput("input-and-output/test-output.txt")
	frequencies = PQ()
	guesses = []

	# search corpus for n-grams
	for head,tail in contexts:
		getNgrams(corpus, frequencies, "its", head, tail)
		getNgrams(corpus, frequencies, "it's", head, tail)
		# also check for "it is" and "it has"
		getNgrams(corpus, frequencies, "it is", head, tail)
		getNgrams(corpus, frequencies, "it has", head, tail)

		# get the top choice
		best = frequencies.get()[1][0]
		if best == "it is" or best == "it has":
			best = "it's"
		guesses.append(best)

	# compute accuracy based on output file
	errors = sum([1 if x != y else 0 for x,y in zip(answers, guesses)])
	accuracy = 1.0 - errors / len(answers)
	print("Accuracy:", accuracy)