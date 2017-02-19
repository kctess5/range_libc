from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

cdef extern from "includes/RangeLib.h":
    # define flags
    cdef bool _USE_CACHED_TRIG "_USE_CACHED_TRIG"
    cdef bool _USE_ALTERNATE_MOD "_USE_ALTERNATE_MOD"
    cdef bool _USE_CACHED_CONSTANTS "_USE_CACHED_CONSTANTS"
    cdef bool _USE_FAST_ROUND "_USE_FAST_ROUND"
    cdef bool _NO_INLINE "_NO_INLINE"
    cdef bool _USE_LRU_CACHE "_USE_LRU_CACHE"
    cdef int  _LRU_CACHE_SIZE "_LRU_CACHE_SIZE"

cdef extern from "includes/RangeLib.h" namespace "ranges":
    cdef cppclass OMap:
        OMap(int w, int h)
        OMap(string filename)
        OMap(string filename, float threshold)
        unsigned width
        unsigned height
        vector[vector[bool]] grid
        bool save(string filename)
        bool error()
        bool get(int x, int y)
    cdef cppclass BresenhamsLine:
        BresenhamsLine(OMap m, float mr)
        float calc_range(float x, float y, float heading)
        void numpy_calc_range(float * ins, float * outs, int num_casts)
    cdef cppclass RayMarching:
        RayMarching(OMap m, float mr)
        float calc_range(float x, float y, float heading)
        void numpy_calc_range(float * ins, float * outs, int num_casts)
    cdef cppclass CDDTCast:
        CDDTCast(OMap m, float mr, unsigned int td)
        float calc_range(float x, float y, float heading)
        void numpy_calc_range(float * ins, float * outs, int num_casts)
    cdef cppclass GiantLUTCast:
        GiantLUTCast(OMap m, float mr, unsigned int td)
        float calc_range(float x, float y, float heading)
        void numpy_calc_range(float * ins, float * outs, int num_casts)

# define flags
USE_CACHED_TRIG = _USE_CACHED_TRIG
USE_ALTERNATE_MOD = _USE_ALTERNATE_MOD
USE_CACHED_CONSTANTS = _USE_CACHED_CONSTANTS
USE_FAST_ROUND = _USE_FAST_ROUND
NO_INLINE = _NO_INLINE
USE_LRU_CACHE = _USE_LRU_CACHE
LRU_CACHE_SIZE = _LRU_CACHE_SIZE


'''
Docs:

PyOMap: wraps OMap class
    constructor: PyOMap(arg1, arg2=None)
        Type options: <type(arg1)> <type(arg2)>
            <int width>, <int height> : empty omap of size width, height
            <string map_path>         : loads map from png image at given path
            <string map_path>, <float>: loads map from png image at given path with given occupancy threshold
            <numpy.ndarray>           : loads map from given numpy boolean array
    methods:
        bool save(string filename)    : saves the occupancy grid to given path in png format. 
                                        white == free, black == occupied
        bool isOccupied(int x, int y) : returns true if the given pixel index is occupied, false otherwise
        bool error()                  : returns true if there was an error loading the map

'''

cdef class PyOMap:
    cdef OMap *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, arg1, arg2=None):
        if arg1 is not None and arg2 is not None:
            if isinstance(arg1, int) and isinstance(arg1, int):
                self.thisptr = new OMap(<int>arg1,<int>arg2)
            else:
                self.thisptr = new OMap(<string>arg1,<float>arg2)
        elif arg1 is not None:
            if isinstance(arg1, np.ndarray):
                height, width = arg1.shape
                self.thisptr = new OMap(<int>height,<int>width)
                for y in xrange(height):
                    for x in xrange(width):
                        self.thisptr.grid[x][y] = <bool>arg1[y,x]
            else:
                self.thisptr = new OMap(arg1)
        else:
            print "Failed to construct PyOMap, check argument types."
            self.thisptr = new OMap(1,1)

    def __dealloc__(self):
        del self.thisptr

    cpdef bool save(self, string fn):
        return self.thisptr.save(fn)

    cpdef bool isOccupied(self, int x, int y):
        return self.thisptr.get(x, y)

    cpdef bool error(self):
        return self.thisptr.error()

    cpdef int width(self):
        return self.thisptr.width

    cpdef int height(self):
        return self.thisptr.height

cdef class PyBresenhamsLine:
    cdef BresenhamsLine *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, PyOMap Map, float max_range):
        self.thisptr = new BresenhamsLine(deref(Map.thisptr), max_range)
    def __dealloc__(self):
        del self.thisptr
    cpdef float calc_range(self, float x, float y, float heading):
        return self.thisptr.calc_range(x, y, heading)
    cpdef void calc_range_np(self,np.ndarray[float, ndim=2, mode="c"] ins, np.ndarray[float, ndim=1, mode="c"] outs):
        self.thisptr.numpy_calc_range(&ins[0,0], &outs[0], outs.shape[0])

cdef class PyRayMarching:
    cdef RayMarching *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, PyOMap Map, float max_range):
        self.thisptr = new RayMarching(deref(Map.thisptr), max_range)
    def __dealloc__(self):
        del self.thisptr
    cpdef float calc_range(self, float x, float y, float heading):
        return self.thisptr.calc_range(x, y, heading)
    cpdef void calc_range_np(self,np.ndarray[float, ndim=2, mode="c"] ins, np.ndarray[float, ndim=1, mode="c"] outs):
        self.thisptr.numpy_calc_range(&ins[0,0], &outs[0], outs.shape[0])

cdef class PyCDDTCast:
    cdef CDDTCast *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, PyOMap Map, float max_range, unsigned int theta_disc):
        self.thisptr = new CDDTCast(deref(Map.thisptr), max_range, theta_disc)
    def __dealloc__(self):
        del self.thisptr
    cpdef float calc_range(self, float x, float y, float heading):
        return self.thisptr.calc_range(x, y, heading)
    cpdef void calc_range_np(self,np.ndarray[float, ndim=2, mode="c"] ins, np.ndarray[float, ndim=1, mode="c"] outs):
        self.thisptr.numpy_calc_range(&ins[0,0], &outs[0], outs.shape[0])

cdef class PyGiantLUTCast:
    cdef GiantLUTCast *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, PyOMap Map, float max_range, unsigned int theta_disc):
        self.thisptr = new GiantLUTCast(deref(Map.thisptr), max_range, theta_disc)
    def __dealloc__(self):
        del self.thisptr
    cpdef float calc_range(self, float x, float y, float heading):
        return self.thisptr.calc_range(x, y, heading)
    cpdef void calc_range_np(self,np.ndarray[float, ndim=2, mode="c"] ins, np.ndarray[float, ndim=1, mode="c"] outs):
        self.thisptr.numpy_calc_range(&ins[0,0], &outs[0], outs.shape[0])

cdef class PyNull:
    def __cinit__(self, PyOMap Map, float max_range, unsigned int theta_disc):
        pass
    def __dealloc__(self):
        pass
    cpdef float calc_range(self, float x, float y, float heading):
        return x + y + heading
    cpdef void calc_range_np(self,np.ndarray[float, ndim=2, mode="c"] ins, np.ndarray[float, ndim=1, mode="c"] outs):
        a = ins[0,0]
        b = outs[0]
        c = outs.shape[0]