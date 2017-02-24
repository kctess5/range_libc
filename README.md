# RangeLibc

This library provides for different implementations of 2D raycasting for 2D occupancy grids. The code is written and optimized in C++, and Python wrappers are also provided.

## Building the Code

The following has been tested on both Ubuntu 14.04 and OSX 10.10, hopefully it will work on other systems as well, or will at least be not too difficult to fix.

### C++ code

```
# clone the repository
git clone https://github.mit.edu/chwalsh/range_libc_dist
cd range_libc_dist
mkdir build
cd build
cmake ..
make
```

If you get an error about having the wrong version of CMake, install a version of CMake that is >= version 3.3 from here: https://cmake.org/install/

If you don't want to update your system's version of CMake, simply:

```
# unzip cmake download and cd into that directory
mkdir build
cd build
cmake ..
make
# 3.9 should be your cmake version number
sudo ln -s [path to cmake directory]/build/bin/cmake /usr/bin/cmake3.9
```

Then use cmake3.9 instead of cmake in the above instructions for building the range_lib code.

### Python Wrappers

To build the code and its associated Python wrappers for use in Python code, do the following. You may have to install Cython if you do not already have it on your system.

```
# clone the repository
git clone https://github.mit.edu/chwalsh/range_libc_dist
cd range_libc_dist/pywrapper
# for an in place build, do this:
python setup.py build_ext --inplace
# for a system wide install, do this:
python setup.py install
# this should take a few seconds to run
python test.py
```

Take a look at test.py in the pywrapper directory for example usage. It is recommended that you use the calc_range_np method with batched queries, as it is significantly faster due to lower function call overhead per query. Basically, you simply populate a Numpy array with the (x,y,theta) queries and the function will populate a provided numpy array with the results. Under the hood, the code operates directly on the Numpy data structure, eliminating the need to copy data back and forth through function calls.

### Building on a RACECAR

MIT's 6.141 uses this library for accelerating particle filters onboard the RACECAR platform. To install this on the Jetson TX1, do:

```
# Copy the code
cd range_libc_dist 
mkdir build
cmake ..
make
# To build the Python wrappers
sudo apt-get install Cython
cd pywrapper
python setup.py install
```

## License

This code is licensed under Apache 2.0. Copyright 2017 Corey H. Walsh. 

You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Enjoy!

## Code structure

```
range_libc_dist/
├── build
│   └── bin          # this is where compiled binaries will be placed
├── CMakeLists.txt   # compilation rules - includes, etc
├── includes
│   ├── lru_cache.h  # implementation of LRU_cache, optionally used
│   ├── RangeLib.h   # main RangeLib source code
│   └── RangeUtils.h # various utility functions
├── license.txt
├── main.cpp         # example c++ usage and simple benchmarks
├── make_plots.py    # turns fine-grained benchmark information into violin plots
├── tmp/             # make this directory for saving fine-grained timing information
├── maps             # example PNG maps
│   └── [various .png files]
├── pywrapper
│   ├── RangeLib.pyx # wrapper file for using RangeLib from Python
│   ├── setup.py     # compilation rules for Cython
│   └── test.py      # example Python usage
├── README.md
└── vendor           # various dependencies, see in here for licenses
    ├── distance_transform # for computing euclidean distance transform
    ├── gflags       # command line flag library from Google
    └── lodepng      # simple PNG loading/saving library
```

## RangeLibc Algorithms Overview

<!--- ///////////////////////////// Bresenham's Line Description ////////////////////////////// -->

### Bresenham's Line

<!--- /////////////////////////////// Ray Marching Description //////////////////////////////// -->
### Ray Marching

Ray marching is a well known algorithm, frequently used to accelerate fractal or volumetric graphical rendering applications. The basic idea can be understood very intuitively. Imagine that you are in an unknown environment, with a blindfold on. If an oracle tells you the distance to the nearest obstacle, you can surely move in any direction by at most that distance without colliding with any obstacle. By applying this concept recursively, one can step along a particular ray by the minimum distance to the nearest obstacle until colliding with some obstacle.

In the occupancy grid world, it is possible to precompute the distance to the nearest obstacle for every discrete state in the grid via the euclidean distance transform. 

#### Pseudocode:

```
# compute the distance transform of the map
def precomputation(omap):
	distance_transform = euclidean_dt(omap)

# step along the (x,y,theta) ray until colliding with an obstacle
def calc_range(x,y,theta):
	t = 0.0
	coeff = 0.99
	while t < max_range:
		px, py = x + cos(theta) * t, y + sin(theta) * t

		if px or py out of map bounds:
			return max_range

		dist = distance_transform[px,py]
		if dist == 0.0:
			return sqrt((x - px)^2 + (y - py)^2)

		t += max(dist*coeff, 1.0)

```

#### Analysis

Precomputation: O(|theta_discretization|\*|edge pixels in occupancy grid|+|theta_discretization|\*|occupied pixels|\*log(|occupied pixels|))
Pruning: O(|map width|\*|map height|\*|theta_discretization|\*log(min(|occupied pixels|, longest map dimension)))
Calc range: O(log(min(|occupied pixels|, longest map dimension)))
Memory: O(|theta_discretization|\*|edge pixels|) - in practice much smaller, due to pruning

** Pros **

- Fast calc_range on average
- Space efficent
- Easy to implement
- Fairly fast to compute distance transform
- Extends easily to 3D

** Cons **

- Poor worst case performance - degenerate case similar to Bresenham's line
- High degree of control flow divergence (for parallelization)
- Not great for incrementally changing the map

<!--- /////////////////////////////////// CDDT Description //////////////////////////////////// -->

### Compressed Directional Distance Transform (ours)

Create a data structure which allows the distance value to be computed for any discrete (x,y,theta) ray in fast (but not constant) time. The easiest way to visualize this datastructure is to imagine what I refer to as a directional distance transform. A euclidean distance transform stores the distance to the nearest obstacle *in any direction* for every discrete state in a grid. A directional distance transform instead stores the distance to the nearest obstacle in a particular direction theta for each state in the grid. Given a directional distance transform for a particular theta, one can ascertain the position of the nearest obstacle in the theta direction from each discrete position in constant time, simply reading from the table. By precomputing the directional distance transform for every discrete theta, constant time ray casting is possible for any (x,y,theta) state. The obvious downside is that such a table would be very large - storing |x|*|y|*|theta| discrete states would require gigabytes of memory for reasonably large maps.

<img alt="Distance Transform examples" src="./graphics/dist_figure.png">
<p align="center" style="color:gray; margin-top: 0">Fig 1</p>

A key observation is that such a table would contain a large amount of redundant information, which can be easily compressed. This can be seen in the case where theta is some cardinal direction, so the distance value for adjacent grid cells in one direction can be modeled as a sawtooth wave with a slope of 1. By storing only the position of where the sawtooth wave goes to zero, the correct distance value can be attained for any cell in that row. The zero points are simply the position of all the obstacles in that row. So for each theta value, all obstacles in the source map are projected onto one axis, and for some discrete set of bins, the position of all obstcles falling in that bin are stored.

Clearly, this would create theta copies of the geometry from the source map, so the memory requirement is now |theta|*|occupied grid cells| which is at worst |x|*|y|*|theta|. However, the memory requirement can be further reduced by a factor of 2 by exploiting the rotational symmetry. The projection along the 0 direction is just the reverse of the projection along the pi direction. Thus, one only needs to store |0:pi| compressed LUTs in order to span the theta space.

Furthermore, several optimizations exist when one considers which obstacles in the grid will never result in a ray casting collision. For example, consider a 3x3 block of obstacles in an otherwise empty map. The center obstcle will never be the nearest neighbor in any ray casting query, because any such query would first intersect with one of the 8 surrounding obstacles. To exploit this, one can take the morphological edge map of the source map and use that to generate the compressed LUT datastructure without loss of generality. This can result in significant memory usage reductions in dense maps, where the number of obstacles adjacent to empty cells is small compared to the total number of obstacles. 

Similarly, we can prune many of the obstacles in a given theta bin that could not possibly result in a collision. Consider the ca se where a colinear wall is entirely projected into a single theta bin. Along the theta direction implied by the theta bin, only one point can possibly be involved in a ray casting collision - the edgemost obstacle. Since we can ray cast in two directions for each theta bin, there are in fact two edgemost obstacles that must be considered. All non-edge wall components may be safely discarded without loss of generality. This is easy to understand in the cardinal directions, where pixel boundries line up nicely, but in the non-cardinal directions it becomes more difficult to determine which obstacles could possibly be involved in a collision. Rather than attempting to analytically determine which obstacles will result in ray casting collisions, it is easiest to ray cast from every possible (x,y,theta) and prune each LUT element which is never involved in a collision. 

#### Pseudocode:

```

# for the given theta, determine a translation that will ensure the 
# y coordinate of every pixel in the rotated map will be positive
def y_offset(theta):
	pass

# give the range of y coordinates that the pixel overlaps with
def y_bounds(pixel):
	return range(min(pixel.corners.y), max(pixel.corners.y))

# build the CDDT datastructure
def precomputation(omap):
	# prune any unimportant geometry from the map
	edgeMap = morphological_edge_transform(omap)

	# build the empty LUT data structure
	compressed_lut = []
	for each theta in |theta_discretization|:
		projection_lut = []
		for each i in range(lut_widths[theta]):
			projection_lut.append([])
		compressed_lut.append(projection_lut)

	# populate the LUT data structure
	for each theta in |theta_discretization|:
		for each occupied pixel (x,y) in omap:
			pixel.rotate(theta)
			pixel.translate(y_offset(theta))
			lut_indices = y_bounds(pixel)
			
			for each index in lut_indices:
				compressed_lut[theta][index].append(pixel.center.x)

	# sort each LUT bin for faster access via binary search
	for each theta in |theta_discretization|:
		for each i in compressed_lut[theta].size():
			sort(compressed_lut[theta][i])

# (optional) remove unused entries from the LUT to save space
# highly recommended for static maps
def prune():
	# build an empty table of sets to keep track of which
	# indices in the CDDT data structure are used
	collision_table = []
	for theta in range(theta_discretization):
		collision_row = []
		for i in range(compressed_lut[theta].size()):
			collision_row.append(set())
		collision_table.append(collision_row)

	# ray cast from every possible (x,y,theta) state, keeping track
	# of which LUT entries are used
	for x in range(omap.width):
		for y in range(omap.height):
			for theta in range(theta_discretization):
				# keep track of which object in the LUT is found to collide
				# with the following ray cast query
				calc_range(x,y,theta) implies (lut_bin, lut_bin_index)
				collision_table[theta][lut_bin].add(lut_bin_index)

	# remove any element of the LUT that is not in the collision table
	for theta in range(theta_discretization):
		for i in range(compressed_lut[theta].size()):
			new_lut_bin = []
			for obstacle in compressed_lut[theta][i]:
				if obstacle in collision_table:
					new_lut_bin.append(obstacle)
				else: continue
			compressed_lut[theta][i] = new_lut_bin

# compute the distance to the nearest obstacle in the (x,y,theta) direction
def calc_range(x,y,theta):
	angle_index, discrete_angle, flipped_search = discretize_theta(theta)
	lut_x, lut_y = rotate(x, y, discrete_angle)

	if omap.occupied(x,y):
		return 0.0

	lut_bin = compressed_lut[angle_index][(int)lut_y]
	if flipped_search:
		nearest_obstacle_x = lut_bin.next_lesser_element(lut_x)
	else:
		nearest_obstacle_x = lut_bin.next_greater_element(lut_x)
	
	distance = abs(nearest_obstacle_x - lut_x)
	return distance
```

#### Analysis

Precomputation: O(|width|*|height|) for 2D grid. In general O(dk) where d is the dimensionality of the grid, and k is the number of grid locations.
Calc range: worst case O(|longest map dimension|), on average much faster (close to logarithmic performance in scene size)
Memory: O(|width|*|height|) for 2D grid. In general O(k) where k is the number of grid locations.

** Pros **

- Fast calc_range, in practice nearly constant time
- Radial symmetry optimizations can provide additional speed in the right context
- Potential for online incremental compressed LUT modification for use in SLAM (would need to store additional metadata)
- Space efficent

** Cons **

- Slow construction and pruning time
- Approximate due to the discrete theta space
- Can be difficult to implement well
- Curse of dimensionality in higher dimensions


<!--- //////////////////////////////////// LUT Description //////////////////////////////////// -->

### Giant Lookup Table

Precompute distances for all possible (x,y,theta) states.

#### Pseudocode:

```
# For every (x,y,theta) in a predefined grid, use Besenham's line or 
# ray maching to build the table
def precomputation(omap):
	giant_LUT[width][height][theta_discretization] = -1
	for x in range(omap.width):
		for y in range(omap.height):
			for theta in range(theta_discretization):
				giant_LUT[x][y][theta] = calc_range(x,y,theta)

# simply read from the table
# note: interpolation between the two closest discrete 
#       thetas would be more accurate but slower
def calc_range(x,y,theta):
	return giant_LUT[int(x), int(y), discrete(theta)]
```

#### Analysis

Precomputation: O(|theta_discretization|\*|width|\*|height|\*O(calc_range))
Memory: O(|theta_discretization|\*|width|\*|height|)
Calc range: O(1)

** Pros **

- Very fast calc_range
- Easy to implement

** Cons **

- Very slow construction time
- Approximate due to the discrete theta space
- Very large memory requirement
- Curse of dimensionality in higher dimensions


### Notes
#### Interpolation


### Benchmarks

### Citations

http://people.cs.uchicago.edu/~pff/papers/dt.pdf