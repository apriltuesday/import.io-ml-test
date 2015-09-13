# April Shen -- 2015-09-09
# Extra classes for grade prediction -- Datum and Decision Stump
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


class DecisionStump:
	"""
	A decision tree with only one node (the root), trained by minimizing
	the entropy of the split.
	"""

	# numValues is both the number of possible feature values
	# and the number of possible class labels
	def __init__(self, numValues):
		# feature at the root
		self.root = 0
		# class label for each possible value, chosen initially to be in the middle
		# of the possible class outputs
		self.labels = [int(numValues/2.0)] * (numValues+1) # +1 to account for missing subject


	def classify(self, d):
		"""
		Return classification of d, based on its value of the root feature.
		"""
		return self.labels[d.features[self.root]+1]


	def splitEntropy(self, split, classLabels, featureVals):
		"""
		Compute entropy of this split. Split entropy is the sum of the entropy
		in class labels on each branch, multiplied by the number of instances
		on that branch.

		split: list of Counters, one for each possible feature value,
		which count the number of data instances with a particular class
		label for that branch of the tree.
		classLabels: list of possible class values
		featureVals: list of possible feature values (including -1 for missing)
		"""
		# note that all data is weighted here
		total = 0.0
		for j in featureVals:
			# N_j is the total number of instances that are on branch j
			N_j = sum([x[1] for x in split[j].items()])
			if N_j == 0:
				continue

			# compute entropy of branch j
			entropy = 0.0
			for i in classLabels:
				# p_ij is the probability that an instance taking branch j has class i
				p_ij = float(split[j][i]) / N_j
				if p_ij != 0:
					# compute sum of entropy for each 
					entropy += p_ij * log(p_ij)
					
			# split entropy is entropy of each branch weighted by proportion of data
			# in that branch
			total += N_j * entropy
		return -total


	def train(self, data, numFeatures):
		"""
		Train on the weighted data, given the number of features.
		Training is done by splitting on the feature that minimizes
		split entropy, then assigning the majority label to each branch.
		"""
		minEnt = float("inf")
		minEntSplit = None
		minEntFeat = 0
		classLabels = range(len(self.labels))
		#-1 when feature is absent, otherwise possible feature values same as possible class labels
		featureVals = range(-1, len(self.labels))

		for f in range(numFeatures):
			# split is a list of groups for each possible value of f.
			# split[f] counts the number of data points with each class
			# label for this value of f.
			split = [Counter() for x in featureVals]
			for d in data:
				# note these are weighted counts
				# +1 to account for missing feature
				split[d.features[f]+1][d.label] += d.weight 

			# compute entropy of this split
			entropy = self.splitEntropy(split, classLabels, featureVals)
			if entropy < minEnt:
				minEnt = entropy
				minEntSplit = split
				minEntFeat = f

		# choose the split that minimizes entropy
		self.root = minEntFeat
		# for each branch of the best split, assign the majority label
		for i in featureVals:
			if len(minEntSplit[i]) != 0:
				biggest =  max(minEntSplit[i].items(), key=lambda x: x[1])
				self.labels[i] = biggest[0]