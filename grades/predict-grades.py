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

# list of data
data = []


############### ADABOOST ###################


def initializeWeights():
    """
    Initialize all weights.
    """
    w = 1.0 / len(data)
    for d in data:
        d.weight = w


def sampleData(n):
    """
    Returns a list of n data instances sampled from data according to
    the distribution of weights.
    """
    sampleList = []
    for i in range(n):
        k = random()
        runningSum = 0.0
        for d in data:
            if d in sampleList: # sampling without replacement
                continue
            runningSum += d.weight
            if runningSum > k:
                sampleList.append(d)
                break
    return sampleList


def reweight(predictions, error):
    """
    Reweighting. See Schapire 1999.
    """
    # XXX Fix reweighting....
    # error of this ensemble
    #error = 1 - accuracy
    # perturb by epsilon to prevent getting log(0)
    if error == 1:
        error -= 0.00001
    elif error == 0: # did perfectly, don't reweight
        return
    # importance factor
    alpha = 0.5 * log((1 - error) / error)
    # normalization factor (running sum)
    z_t = 0.0
    # update weights (numerator)
    for d, p in zip(data, predictions):
        if p == d.label:
            #alpha *= -1
            # if correctly labeled, set weights to 0 XXXX
            d.weight = 0.0
            continue
        d.weight = d.weight * exp(alpha)
        z_t += d.weight
    if z_t == 0:
        return
    # normalize, to get a probability distribution
    for d in data:
        d.weight /= z_t
    # return this model's weight
    return alpha


###################### MAIN ######################


# parse JSON input (either training or test)
def parseInput(filename):
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
		if label:
			d.label = label
		data.append(d)
	


# parse test output file
def parseOutput(filename):
	content = open(filename).readlines()
	for l, d in zip(content, data):
		d.label = int(l)


# train stump to classify based on majority class for each
# value of its root features
def trainStump(stump, sample):
	groups = [Counter() for x in range(D)]
	for d in sample:
		groups[d.features[stump.root]][d.label] += 1
	for i in range(D):
		if len(groups[i]) != 0:
			biggest =  max(groups[i].items(), key=lambda x: x[1])
			stump.classes[i] = biggest[0]


# train ensemble of decision stumps using AdaBoost
# return list of (m, a) where m is the base model and a is its weight
def train():
	ensemble = []
	initializeWeights()

	for i in range(D):
		# new base learner
		stump = DecisionStump(i, 8) #8 is number of possible grades
		# sample
		sample = sampleData(10) # XXX how many samples?
		# train base learner on sample
		trainStump(stump, sample)
		# predict for all data
		predictions = [stump.classify(d) for d in data]
		# calculate error
		error = sum([d.weight * (1 if d.label != p else 0) for d,p in zip(data, predictions)])
		# reweight
		alpha = reweight(predictions, error)
		# add to ensemble
		ensemble.append((stump, alpha))

	return ensemble

	

def test(ensemble):
	pass

if __name__ == '__main__':
	parseInput("small/training.json")
	ensemble = train()
	for stump, alpha in ensemble:
		print(alpha, stump.root, stump.classes)