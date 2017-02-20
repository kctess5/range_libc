from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	ext_modules=[
		Extension("range_lib", ["RangeLib.pyx","../vendor/lodepng/lodepng.cpp"], 
			extra_compile_args = ["-w","-std=c++11", "-march=native", "-ffast-math", "-fno-math-errno", "-O3"],
			include_dirs = [
				"../",
				"../vendor/distance_transform/",
				"../vendor/distance_transform/include",
				"../vendor/distance_transform/extern/DopeVector/include/"
			],
			depends=["../includes/*.h"],
			language="c++",)],
    cmdclass = {'build_ext': build_ext})