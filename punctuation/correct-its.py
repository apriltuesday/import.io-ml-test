# April Shen -- 2015-09-11
# The Punctuation Corrector (https://www.hackerrank.com/challenges/punctuation-corrector-its)
import nltk, re
from nltk.corpus import gutenberg, webtext, brown, inaugural
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
    for i in range(0,hLength+1):
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
       

# Searches corpora for number of instances of an ngram, returns count
def searchCorpus(corpus, ngram):
    words = " ".join(ngram)
    count = corpus.count(words)
    return count


def readCorpus():
	f =  open("corpus.txt", encoding="utf-8")
	corpus = "".join(f.readlines()).lower()
	# add some nltk corpora
	# brown corpus contains annotations, which we remove
	browntext = brown.raw()
	browntext = re.sub(r'/.*?\s',' ',browntext)
	corpus += gutenberg.raw() + "\n" + webtext.raw() + "\n" + browntext  + "\n" + inaugural.raw()
	return corpus


def parseInput(filename):
	# return list of heads and tails, divided by ???
	delimiter = "???"
	parsed = []
	with open(filename) as f:
		for line in f.readlines()[1:]:
			i = line.index(delimiter)
			parsed.append(tuple(line.split(delimiter)))
	return parsed


def parseOutput():
	pass


if __name__ == "__main__":
	corpus = readCorpus()
	contexts = parseInput("input-and-output/sample-input.txt")
	frequencies = PQ()

	# XXX handle multiple ??? per sentence
	for head,tail in contexts:
		getNgrams(corpus, frequencies, "its", head, tail)
		getNgrams(corpus, frequencies, "it's", head, tail)

		# top choice
		# XXX print results, compute accuracy based on output file
		print(frequencies.get()[1])