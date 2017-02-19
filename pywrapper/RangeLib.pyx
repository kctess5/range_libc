cdef extern from "includes/RangeLib.h" namespace "ranges":
    # cdef cppclass OMap:
    #     bool has_error
    #     unsigned width
    #     unsigned height
    cdef cppclass BresenhamsLine:
        BresenhamsLine()
        float calc_range(int x)

cdef class PyBresenhamsLine:
    cdef BresenhamsLine *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        pass
        # self.thisptr = new BresenhamsLine()
    def __dealloc__(self):
        pass
        # del self.thisptr
    # def calc_range(self, x):
    #     return self.thisptr.calc_range(x)