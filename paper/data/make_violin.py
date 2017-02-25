import csv
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def reject_outliers(data, m = 12.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def get_data(fn):
	# print fn
	reader = csv.reader(open(fn,"rb"),delimiter=',')

	data_header = next(reader)

	if len(data_header) == 1:
		data = np.array(list(reader)).astype('float')
		# anything longer than this is an outlier, possibly because of a kernel interupt
		data = data[data<0.0000040]
	else:
		data = np.array(list(reader)).astype('float')[:,3]#*1000000000.0
		# anything longer than this is an outlier, possibly because of a kernel interupt
		data = data[data<0.0000040]


	# raw_data = list(reader)
	# # print raw_data[0], ra
	# result=np.array(raw_data[1:]).astype('float')
	# data = result[:,3]
	# # data = reject_outliers(data)
	# data = data[data<0.0000035]
	data *= 1000 # convert to millis for better rendering
	return data

def violin_plot(path):
	files = ["glt.csv", "pcddt.csv", "cddt.csv", "rm.csv", "bl.csv"]
	label = ['LUT','PCDDT','CDDT','RM',"BL"]
	files = map(lambda x: path + x, files)
	# print files
	data = map(get_data, files)

	pos = [1,2,3,4,5]

	# for i in xrange(5):
	# 	print files[i]
	# 	print np.mean(data[i])
	# 	print np.median(data[i])
	# 	print np.std(data[i])
	# 	pass

	# data = get_data("../range_libc/info/logs/1_lut.csv")
	# pos = [1]

	# reader = csv.reader(open("../range_libc/somefile.csv","rb"),delimiter=',')
	# raw_data = list(reader)
	# print raw_data[0]
	# result=np.array(raw_data[1:]).astype('float')
	# data = reject_outliers(result[:,3])
	# print np.mean(result[:,3])

	fig, axes = plt.subplots(1,1)

	# # pos = [1, 2, 4, 5, 7, 8]
	# # data = [np.random.normal(size=100) for i in pos]

	violin_parts = axes.violinplot(data, pos, points=1000, widths=0.75, vert=False,
                      showmeans=True, showextrema=False, showmedians=True)

	axes.set_yticks(pos)
	axes.set_yticklabels(label)

	axes.set_title("Completion times (ms) for synthetic map, random sampling")
	plt.tight_layout()

	for pc in violin_parts['bodies']:
		pc.set_facecolor('red')
		pc.set_edgecolor('black')
	pass


if __name__ == '__main__':
	print "Hello world."
	# dist_transform = make_small_distance_transform_plot("./maps/basement.map.yaml")
	violin_plot("./synthetic/random/trial3/")
	print "Done making plot"
	# plt.savefig('test.png', bbox_inches='tight')
	plt.show()