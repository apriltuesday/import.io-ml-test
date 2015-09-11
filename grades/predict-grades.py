# April Shen -- 2015-09-09
# Predict the Missing Grade (https://www.hackerrank.com/challenges/predict-missing-grade)

import json
from math import log, exp
from random import random
from collections import Counter
from datum import Datum
from stump import DecisionStump

# map feature string to index
features = {
	"English": 0,
	"Physics": 1,
	"Chemistry": 2,
	"ComputerScience": 3,
	"Biology": 4,
	"PhysicalEducation": 5,
	"Economics": 6,
	"Accountancy": 7,
	"BusinessStudies": 8
}
D = len(features)
K = 8 # number of grade levels (classes for classification)
classes = range(K) #note 0 corr. to 1

# list of data
data = []


############### ADABOOST ###################


def initializeWeights():
    """
    Initialize all weights to be uniform.
    """
    w = 1.0 / len(data)
    for d in data:
        d.weight = w


def reweight(predictions, error):
    """
    Reweighting. See Schapire 1999.
    """
    # perturb by epsilon to prevent getting log(0)
    if error == 1:
        error -= 0.00001
    elif error == 0: # did perfectly, don't reweight
        return
    # importance factor
    alpha = log((1.0 - error) / error) + log(K - 1.0)
    # normalization factor (running sum)
    z_t = 0.0
    # update weights (numerator)
    for d, p in zip(data, predictions):
        d.weight *= exp(alpha * (1 if p != d.label else 0))
        z_t += d.weight
    # normalize, to get a probability distribution
    for d in data:
        d.weight /= z_t
    # return this model's weight
    return alpha


###################### MAIN ######################


def parseInput(filename):
	"""
	Parse JSON input (either training or test) and store in data.
	Stores math grade as label, if present.
	"""
	content = open(filename).readlines()
	for line in content[1:]:
		record = json.loads(line)
		x = [0] * D
		label = None

		for subject in record.keys():
			if subject == "Mathematics":
				#label
				label = record[subject]
			elif subject != "serial":
				#features
				x[features[subject]] = record[subject] - 1

		d = Datum(x)
		if label: #only training data is labeled
			d.label = label - 1
		data.append(d)


def parseOutput(filename):
	"""
	Parse the test output JSON, and store as data labels.
	"""
	content = open(filename).readlines()
	for l, d in zip(content, data):
		d.label = int(l) - 1


def splitEntropy(split):
	"""
	Compute entropy of this split
	"""
	total = 0.0
	# XXX does this deal with weighted data appropriately?
	for j in classes: #note in this case, set of classes and set of feature values are the same
		N_j = sum([x[1] for x in split[j].items()])
		if N_j == 0:
			continue
		partial = 0.0
		for i in classes:
			p_ij = float(split[j][i]) / N_j
			if p_ij != 0:
				partial += p_ij * log(p_ij)
		total += N_j * partial
	return -total


def trainStump():
	"""
	Train one decision stump on the weighted data.
	"""
	# for each feature, measure the entropy of the split
	# split on the feature that minimizes this
	minEnt = float("inf")
	minEntSplit = None
	minEntFeat = 0
	for f in range(D):
		# each group counts the number of data points with a particular
		# label for a particular value of this feature
		split = [Counter() for x in classes]
		for d in data:
			split[d.features[f]][d.label] += d.weight

		#compute entropy of this split
		entropy = splitEntropy(split)
		if entropy < minEnt:
			minEnt = entropy
			minEntSplit = split
			minEntFeat = f

	stump = DecisionStump(minEntFeat, K)
	# for each branch of the best split, assign the majority label
	for i in classes:
		if len(split[i]) != 0:
			biggest =  max(split[i].items(), key=lambda x: x[1])
			stump.classes[i] = biggest[0]
	return stump


def weightedError(predictions):
	"""
	Calculate weighted error of predictions.
	"""
	return sum([d.weight * (1 if d.label != p else 0) for d,p in zip(data, predictions)])


def train(numModels):
	"""
	Train ensemble of numModels decision stumps using (multiclass)
	AdaBoost.
	Return list of (m, a) where m is the decision stump and a is its
	weight in the final ensemble.
	"""
	ensemble = []
	initializeWeights()

	for i in range(numModels):
		# train new base learner
		stump = trainStump()
		# make predictions and comput error for this learner
		predictions = [stump.classify(d) for d in data]
		error = weightedError(predictions)
		# reweight
		alpha = reweight(predictions, error)
		# add to ensemble
		ensemble.append((stump, alpha))

	return ensemble


def test(ensemble):
	"""
	Test performance of ensemble on data, returning
	classification accuracy.
	"""
	initializeWeights()
	predictions = []
	for d in data:
		probs = []
		for k in classes:
			probs.append(sum([(a if m.classify(d) == k else 0) for m,a in ensemble]))
		predictions.append(max(classes, key = lambda i : probs[i]))
	return 1.0 - weightedError(predictions)


if __name__ == '__main__':
	#parseInput("small/training.json")
	parseInput("training-and-test/training.json")
	ensemble = train(10)
	data = []
	#parseInput("small/test.in.json")
	#parseOutput("small/test.out.json")
	parseInput("training-and-test/sample-test.in.json")
	parseOutput("training-and-test/sample-test.out.json")
	accuracy = test(ensemble)
	for stump, alpha in ensemble:
		print(alpha, stump.root, stump.classes)
	print(accuracy)