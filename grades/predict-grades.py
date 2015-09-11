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
				x[features[subject]] = record[subject]

		d = Datum(x)
		if label: #only training data is labeled
			d.label = label
		data.append(d)	


def parseOutput(filename):
	"""
	Parse the test output JSON, and store as data labels.
	"""
	content = open(filename).readlines()
	for l, d in zip(content, data):
		d.label = int(l)


# XXX Fix this: should return the stump
# and choose root based on info gain
def trainStump():
	"""
	Train one decision stump on the weighted data.
	"""
	groups = [Counter() for x in range(D)]
	for d in data:
		groups[d.features[stump.root]][d.label] += d.weight
	for i in range(D):
		if len(groups[i]) != 0:
			biggest =  max(groups[i].items(), key=lambda x: x[1])
			stump.classes[i] = biggest[0]


def weightedError(predictions):
	"""
	Calculate weighted error of predictions.
	"""
	return sum([d.weight * (1 if d.label != p else 0) for d,p in zip(data, predictions)])


def train():
	"""
	Train ensemble of decision stumps using (multiclass) AdaBoost.
	Return list of (m, a) where m is the decision stump and a is its
	weight in the final ensemble.
	"""
	ensemble = []
	initializeWeights()

	for i in range(D):
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
		probs = [(a if d.label == m.classify(d) else 0) for m,a in ensemble]
		predictions.append(max(range(1,8), key = lambda i : probs[i]))
	print(predictions)
	return 1.0 - weightedError(predictions)


if __name__ == '__main__':
	parseInput("small/training.json")
	ensemble = train()
	data = []
	parseInput("small/test.in.json")
	parseOutput("small/test.out.json")
	accuracy = test(ensemble)
	for stump, alpha in ensemble:
		print(alpha, stump.root, stump.classes)
	print(accuracy)