# April Shen -- 2015-09-09
# Project Euler problem 67 (https://projecteuler.net/problem=67)

# parse input
filename = "p067_triangle.txt"
triangle = [x.split(" ") for x in open(filename).readlines()]
size = len(triangle)
for i in range(size):
	for j in range(len(triangle[i])):
		triangle[i][j] = int(triangle[i][j])

# DP table
# maxSumTo[i,j] is the maximum path sum from the root to
# the jth node at level i
maxSumTo = [[] for x in range(size)]

# base case: root's max path is just itself
maxSumTo[0].append(triangle[0][0])

for i in range(1, size):
	for j in range(0, i+1):
		# recursive case: max path sum of a node is its own
		# value, plus the largest max path sum to a parent
		p1 = p2 = float("-inf")
		if j != 0:
			p1 = triangle[i][j] + maxSumTo[i-1][j-1]
		if j != i:
			p2 = triangle[i][j] + maxSumTo[i-1][j]
		maxSumTo[i].append(max(p1, p2))

# answer is the maximum over the leaves (i.e. the last column)
print(max(maxSumTo[-1]))