from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	ext_modules=[
		Extension("range_lib", ["RangeLib.pyx"], 
			extra_compile_args = ["-w","-std=c++11", "-march=native", "-ffast-math", "-fno-math-errno"],
			include_dirs = ["../"],
			language="c++",)],
    cmdclass = {'build_ext': build_ext})