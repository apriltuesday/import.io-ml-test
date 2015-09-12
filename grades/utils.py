# April Shen -- 2015-09-09
# Datum and Decision Stump classes
from math import log
from collections import Counter

class Datum:
	"""
	A simple class to represent a data instance, with a feature
	vector, a class label, and a weight.
	"""

	def __init__(self, x):
		self.features = x
		self.label = 0
		self.weight = 0.0


# XXX check the tree code, it seems the fishiest
class DecisionStump:
	"""
	A decision tree with only one node (the root), trained by minimizing
	the entropy of the split.
	"""

	# constructor, numValues is the number of children
	def __init__(self, numValues):
		# feature at the root
		self.root = 0
		# class label for each possible value, chosen initially to be in the middle
		self.classes = [int(numValues/2.0)] * numValues


	def classify(self, d):
		# classification of d, based on its value of the root feature
		return self.classes[d.features[self.root]]


	def splitEntropy(self, split):
		"""
		Compute entropy of this split
		"""
		total = 0.0
		classLabels = range(len(self.classes))
		# XXX does this deal with weighted data appropriately?
		for j in classLabels: #note in this case, set of classes and set of feature values are the same
			N_j = sum([x[1] for x in split[j].items()])
			if N_j == 0:
				continue
			partial = 0.0
			for i in classLabels:
				p_ij = float(split[j][i]) / N_j
				if p_ij != 0:
					partial += p_ij * log(p_ij)
			total += N_j * partial
		return -total


	def train(self, data, numFeatures):
		"""
		Train on the weighted data, given the number of features.
		"""
		# for each feature, measure the entropy of the split
		# split on the feature that minimizes this
		minEnt = float("inf")
		minEntSplit = None
		minEntFeat = 0
		for f in range(numFeatures):
			# each group counts the number of data points with a particular
			# label for a particular value of this feature
			split = [Counter() for x in self.classes]
			for d in data:
				split[d.features[f]][d.label] += d.weight

			#compute entropy of this split
			entropy = self.splitEntropy(split)
			if entropy < minEnt:
				minEnt = entropy
				minEntSplit = split
				minEntFeat = f

		self.root = minEntFeat
		# for each branch of the best split, assign the majority label
		for i in range(len(self.classes)):
			if len(minEntSplit[i]) != 0:
				biggest =  max(minEntSplit[i].items(), key=lambda x: x[1])
				self.classes[i] = biggest[0]