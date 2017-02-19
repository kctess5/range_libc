import range_lib
import numpy as np
import itertools, time

# print range_lib.USE_CACHED_TRIG
# print range_lib.USE_CACHED_TRIG
# print range_lib.USE_ALTERNATE_MOD
# print range_lib.USE_CACHED_CONSTANTS
# print range_lib.USE_FAST_ROUND
# print range_lib.NO_INLINE
# print range_lib.USE_LRU_CACHE
# print range_lib.LRU_CACHE_SIZE


testMap = range_lib.PyOMap("../maps/synthetic.map.png",1)
# testMap.save("./test.png")

num_vals = 1000000

# vals = np.zeros((3,num_vals), dtype=np.float32)
# vals[0,:] = testMap.width()/2.0
# vals[1,:] = testMap.height()/2.0
# vals[2,:] = np.linspace(0,2.0*np.pi, num=num_vals)

vals = np.random.random((3,num_vals)).astype(np.float32)
vals[0,:] *= testMap.width()
vals[1,:] *= testMap.height()
vals[2,:] *= np.pi * 2.0
ranges = np.zeros(num_vals, dtype=np.float32)

test_states = [None]*num_vals
for i in xrange(num_vals):
	test_states[i] = (vals[0,i], vals[1,i], vals[2,i])

bl = range_lib.PyBresenhamsLine(testMap, 500)
rm = range_lib.PyRayMarching(testMap, 500)
cddt = range_lib.PyCDDTCast(testMap, 500, 108)
glt = range_lib.PyGiantLUTCast(testMap, 500, 108)
null = range_lib.PyNull(testMap, 500, 108)

def bench(obj,name):
	print
	start = time.clock()
	obj.calc_range_np(vals, ranges)
	end = time.clock()
	dur_np = end - start
	print name+" np: finished computing", ranges.shape[0], "ranges in", dur_np, "sec"
	start = time.clock()
	ranges_slow = map(lambda x: obj.calc_range(*x), test_states)
	end = time.clock()
	dur = end - start
	print name+": finished computing", ranges.shape[0], "ranges in", dur, "sec"
	print "Numpy speedup:", dur/dur_np

bench(bl, "bl")
bench(rm, "rm")
bench(cddt, "cddt")
bench(glt, "glt")
bench(null, "null")
