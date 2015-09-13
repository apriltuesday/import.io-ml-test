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
		self.labels = [int(numValues/2.0)] * (numValues+1) #+1 to account for missing subject


	def classify(self, d):
		# classification of d, based on its value of the root feature
		return self.labels[d.features[self.root]+1]


	def splitEntropy(self, split, classLabels, featureVals):
		"""
		Compute entropy of this split
		split is a list of Counters, one for each possible feature value,
		which count the number of data instances with a particular class
		label for that branch of the tree.
		"""
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
		"""
		# for each feature, measure the entropy of the split
		# split on the feature that minimizes this
		minEnt = float("inf")
		minEntSplit = None
		minEntFeat = 0
		classLabels = range(len(self.labels))
		#-1 when feature is absent, otherwise possible feature values same as possible class labels
		featureVals = range(-1, len(self.labels))
		for f in range(numFeatures):
			# split is a list of groups for each possible value of f
			# each group counts the number of data points with a particular
			# label for this value of f
			split = [Counter() for x in featureVals]
			for d in data:
				# note these are weighted counts
				# +1 to feature value to account for missing subject
				split[d.features[f]+1][d.label] += d.weight

			#compute entropy of this split
			entropy = self.splitEntropy(split, classLabels, featureVals)
			if entropy < minEnt:
				minEnt = entropy
				minEntSplit = split
				minEntFeat = f

		self.root = minEntFeat
		# for each branch of the best split, assign the majority label
		for i in featureVals:
			if len(minEntSplit[i]) != 0:
				biggest =  max(minEntSplit[i].items(), key=lambda x: x[1])
				self.labels[i] = biggest[0]