# import math
# import numpy as np

# theta_discretization = 16
# x = 10
# y = 1

# angles = 2.0 * np.pi * np.arange(0, 1.0, 1.0 / theta_discretization)

# for i in angles:
# 	print
# 	print "a:", i
# 	# print "h:", abs(x * math.sin(i)) + abs(y * math.cos(i)), 
# 	# print "pos:", x * math.sin(i) + y * math.cos(i)
# 	# print "trans:", max(0, -1.0 * (x * math.sin(i) + y * math.cos(i)))

# 	trans = max(0, -1.0 * (x * math.sin(i) + y * math.cos(i)))
# 	pos = x * math.sin(i) + y * math.cos(i)

# 	print "final:", pos + trans
# 	# print 
# 	# print 

import range_lib

t = range_lib.PyBresenhamsLine()

# import time

# s = time.time()
# for x in xrange(1,1000000):
# 	t.calc_range(x)
# print time.time() - s

# t = test.PyRectangle(1,2,3,4)

# print t.getLength()

# t.move(1,2)
