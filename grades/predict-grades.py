# April Shen -- 2015-09-09
# Predict the Missing Grade (https://www.hackerrank.com/challenges/predict-missing-grade)
import sys, json
from math import log, exp
from utils import Datum, DecisionStump

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
numFeatures = len(features)
numClasses = 8 # number of grade levels (classes for classification)

# list of data
data = []


########################### WEIGHTING ###########################


def initializeWeights():
    """
    Initialize all weights to be uniform.
    """
    w = 1.0 / len(data)
    for d in data:
        d.weight = w


def reweight(predictions, error):
    """
    Reweight data and return the computed importance factor.
    See Zhu et al. 2006 on multiclass AdaBoost.

    predictions: list of predicted class labels
    error: error rate
    """
    # perturb by epsilon to prevent getting log(0)
    if error == 1:
        error -= 0.00001
    elif error == 0: # did perfectly, don't reweight
        return
    # importance factor
    alpha = log((1.0 - error) / error) + log(numClasses - 1.0)
    # normalization factor (running sum)
    z_t = 0.0
    # update weights
    for d, p in zip(data, predictions):
        d.weight *= exp(alpha * (1.0 if p != d.label else 0.0))
        z_t += d.weight
    # normalize, to get a probability distribution
    for d in data:
        d.weight /= z_t
    # return this model's importance factor
    return alpha


######################### FILE PARSING ###########################


def parseInput(filename):
	"""
	Parse JSON input (either training or test) and store in data.
	Stores math grade as label, if present.
	"""
	content = open(filename).readlines()
	for line in content[1:]:
		record = json.loads(line)
		x = [-1] * numFeatures #-1 indicates subject not present
		label = None

		for subject in record.keys():
			if subject == "Mathematics": # label
				label = record[subject]
			elif subject != "serial": # feature
				x[features[subject]] = record[subject] - 1

		d = Datum(x)
		if label: # only training data is labeled
			d.label = label - 1
		data.append(d)


def parseOutput(filename):
	"""
	Parse the test output JSON, and store as data labels.
	"""
	content = open(filename).readlines()
	for l, d in zip(content, data):
		d.label = int(l) - 1


########################### METRICS ############################


def weightedError(predictions):
	"""
	Return weighted error of predictions.
	"""
	return sum([d.weight * (1 if d.label != p else 0) for d,p in zip(data, predictions)])


def score(predictions):
	"""
	Return score as defined on HackerRank.

	Score = 100 * ((C-W)/N)
	Where C = Number of Correct predictions, not more than one grade
	point away from the actual grade point assigned.
	W = Number of wrong (incorrect) predictions and
	N = Total number of records in the input. 
	"""
	C = 0.0
	N = len(data)
	for p, d in zip(predictions, data):
		if abs(p - d.label) <= 1:
			C += 1
	score = 100 * ((C - (N-C)) / N)
	return score


#################### TRAINING AND TEST ########################


def train(numModels):
	"""
	Train ensemble of numModels decision stumps using (multiclass) AdaBoost.
	Return list of (m, a) where m is the decision stump and a is its
	weight in the final ensemble.
	"""
	ensemble = []
	initializeWeights()

	for i in range(numModels):
		# train new base learner
		stump = DecisionStump(numClasses)
		stump.train(data, numFeatures)
		# make predictions and compute error for this learner
		predictions = [stump.classify(d) for d in data]
		error = weightedError(predictions)
		# reweight
		alpha = reweight(predictions, error)
		# add to ensemble
		ensemble.append((stump, alpha))

	return ensemble


def test(ensemble):
	"""
	Test performance of ensemble on data, returning	the predictions.
	"""
	initializeWeights()
	predictions = []
	classes = range(numClasses)
	for d in data:
		# each model makes a weighted vote for a class.
		# choose the class with the majority vote.
		votes = []
		for k in classes:
			votes.append(sum([(a if m.classify(d) == k else 0) for m,a in ensemble]))
		predictions.append(max(classes, key = lambda i : votes[i]))
	return predictions


########################### MAIN ############################


if __name__ == '__main__':
	args = sys.argv
	if len(args) < 2:
		print("Usage: predict-grades.py <number of models>")
		sys.exit()

	# train
	numModels = int(args[1])
	parseInput("training-and-test/training.json")
	ensemble = train(numModels)

	# test
	data = []
	parseInput("training-and-test/sample-test.in.json")
	parseOutput("training-and-test/sample-test.out.json")
	predictions = test(ensemble)

	# print the ensemble, just for curiosity
	print("Ensemble:")
	for stump, alpha in ensemble:
	 	print(alpha, stump.root, stump.labels)
	print()

	# compute accuracy and score
	accuracy = 1.0 - weightedError(predictions)
	score = score(predictions)
	print("Classification accuracy:", accuracy)
	print("HackerRank score:", score)