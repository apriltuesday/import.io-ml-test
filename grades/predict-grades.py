# April Shen -- 2015-09-09
# Predict the Missing Grade (https://www.hackerrank.com/challenges/predict-missing-grade)

import json

# XXX go home and check iif this is right....

############## DECISION STUMP ##############

############### ADABOOST ###################

def initializeWeights(data):
    """
    Initialize all weights.
    """
    w = 1.0 / len(data)
    for d in data:
        d.weight = w

def sampleData(data, n):
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

def reweight(data, predictions, accuracy):
    """
    Reweighting. See Schapire 1999.
    """
    # error of this ensemble
    error = 1 - accuracy
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

###################### MAIN ######################

# parse JSON input (either training or test)
def parseInput(filename):
	content = open(filename).readlines()
	for line in content[1:]:
		record = json.loads(line)
		for subject in record.keys():
			if subject == "Mathematics":
				#label
				print(subject, record[subject])
			elif subject != "serial":
				#features
				print(subject, record[subject])

def train():
	pass

def test():
	pass

if __name__ == '__main__':
	parseInput("small/training.json")