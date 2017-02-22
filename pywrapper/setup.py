from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy, os, platform

if platform.system().lower() == "darwin":
	os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]
	os.environ["CC"] = "c++"

setup(
	ext_modules=[
		Extension("range_lib", ["RangeLib.pyx","../vendor/lodepng/lodepng.cpp"], 
			extra_compile_args = ["-w","-std=c++11", "-march=native", "-ffast-math", "-fno-math-errno", "-O3"],
			extra_link_args = ["-std=c++11"],
			include_dirs = [
				"../",
				"../vendor/distance_transform/",
				"../vendor/distance_transform/include",
				"../vendor/distance_transform/extern/DopeVector/include/",
				numpy.get_include()
			],
			depends=["../includes/*.h"],
			language="c++",)],
    cmdclass = {'build_ext': build_ext})