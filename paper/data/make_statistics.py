
''' Makes a table like this:
Method Name   
-----------------------------------------------------------
CDDT
PCDDT
LUT
Ray Marching
Bresenham's
'''

from tabulate import tabulate
import csv
import scipy
import scipy.stats
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import genfromtxt

BYTES_TO_MB = 1.0 / (1024.0*1024.0)

class Trial(object):
	""" Loads data from a single trial """
	def __init__(self, path,method):
		full_path = path + "/" + method + ".csv"
		# print "Loading", full_path
		self.methods = {}
		self.raw_data = csv.reader(open(full_path,"rb"),delimiter=',')

		self.data_header = next(self.raw_data)

		if len(self.data_header) == 1:
			self.data = np.array(list(self.raw_data)).astype('float')
			# anything longer than this is an outlier, possibly because of a kernel interupt
			self.data = self.data[self.data<0.0000040]
			# self.data = self.data[self.data<0.0001]
		else:
			self.data = np.array(list(self.raw_data)).astype('float')[:,3]#*1000000000.0
			# anything longer than this is an outlier, possibly because of a kernel interupt
			self.data = self.data[self.data<0.0000040]
			# self.data = self.data[self.data<0.0001]

		self.construction_time = None
		self.memory = None
		try:
			self.summary   = list(csv.reader(open(path + "/summary.csv","rb"),delimiter=','))
			self.summary = filter(lambda x: method == x[0], self.summary)[0]

			self.construction_time = float(self.summary[1])
			self.memory = float(self.summary[2])*BYTES_TO_MB
		except:
			pass
	def iqr(self):
		q75, q25 = np.percentile(self.data, [75 ,25])
		return q75 - q25

	def mean(self):
		return np.mean(self.data)

	def median(self):
		return np.median(self.data)

	def hist(self):
		plt.hist(self.data,bins=200)

	def init_time(self):
		return self.construction_time

	def memory(self):
		return self.memory


class BenchmarkData(object):
	""" Loads and provides statistics about runtime characteristics """
	def __init__(self, path, method, num_trials):
		self.trials = [Trial(path + "/trial"+str(num_trials), method)]
	def iqr(self):
		return self.trials[0].iqr()

	def mean(self):
		return self.trials[0].mean()

	def median(self):
		return self.trials[0].median()

	def init_time(self):
		return self.trials[0].construction_time

	def memory(self):
		return self.trials[0].memory

def make_table(path, trials):
	table_data = []
	baseline = -1.0
	for i in xrange(1,trials+1):
		table_data.append(["Trial " +str(i)])
		for name in ["bl", "rm", "cddt", "pcddt", "glt"]:
			data = BenchmarkData(path, name, i)
			mean = data.mean()
			iqr = data.iqr()
			if name == "bl":
				baseline = mean
				speedup = 1.0
			else:
				speedup = baseline / mean

			table_data.append([name, mean,data.median(),iqr, iqr / mean, speedup, data.init_time(),data.memory()])
		table_data.append(["  "])
	
	return tabulate(table_data, ["Method", "Mean", "Median", "IQR", "IQR/Mean", "Speedup",  "Construction Time", "Memory"])
		
if __name__ == '__main__':
	# Trial("./","test")
	print "\nBasement Map, Random Sampling"
	print make_table("./basement/random",3)
	print "\nBasement Map, Grid Sampling"
	print make_table("./basement/grid",3)
	print "\nBasement Map, Particle Filter"
	print make_table("./basement/particle",3)

	print "\nSynthetic Map, Random Sampling"
	print make_table("./synthetic/random",3)
	print "\nSynthetic Map, Grid Sampling"
	print make_table("./synthetic/grid",3)
	# print "\nSynthetic Map, Particle Filter"
	# print make_table("./synthetic/particle",1)

	# table_data = []
	# for i in xrange(1,4):
	# 	table_data.append(["Trial " +str(i)])
	# 	for name in ["bl", "rm", "cddt", "pcddt", "glt"]:
	# 		data = BenchmarkData("./synthetic/random", name, i)
	# 		mean = data.mean()
	# 		iqr = data.iqr()
	# 		if name == "bl":
	# 			baseline = mean
	# 			speedup = 1.0
	# 		else:
	# 			speedup = baseline / mean

	# 		table_data.append([name, mean,data.median(),iqr, iqr / mean, speedup, data.init_time(),data.memory()])
	# 	table_data.append(["  "])
	
	# print "Basement Map, Random Sampling"
	# print tabulate(table_data, ["Method", "Mean", "Median", "IQR", "IQR/Mean", "Speedup",  "Construction Time", "Memory (MB)"])