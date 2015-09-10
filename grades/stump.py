# April Shen -- 2015-09-09
# Decision stump

class DecisionStump:

	def __init__(self, root, numValues):
		self.root = root #root feature index
		# class label for each possible value
		# default value is in the middle
		self.classes = [int((numValues+1)/2.0)] * (numValues+1)

	# classification based on majority class
	def classify(self, d):
		return self.classes[d.features[self.root]]