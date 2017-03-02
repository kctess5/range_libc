/*
* Copyright 2017 Corey H. Walsh (corey.walsh11@gmail.com)

* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

*     http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
Useful Links: https://github.com/MRPT/mrpt/blob/4137046479222f3a71b5c00aee1d5fa8c04017f2/libs/slam/include/mrpt/slam/PF_implementations.h
	
	- collision avoidance http://users.isy.liu.se/en/rt/fredrik/reports/01SPpf4pos.pdf
	- σMCL http://www.roboticsproceedings.org/rss01/p49.pdf
	- nifty heuristic to figure out when localization is lost
			https://www.deutsche-digitale-bibliothek.de/binary/6WAGERZFR4H4PREZXILJRER6N7XDVX3H/full/1.pdf
	- http://www.cs.cmu.edu/~16831-f14/notes/F11/16831_lecture04_tianyul.pdf
	- https://april.eecs.umich.edu/pdfs/olson2009icra.pdf
*/

#ifndef RANGE_LIB_H
#define RANGE_LIB_H

#include "vendor/lodepng/lodepng.h"
// #include "vendor/dt/dt.h"

#include "includes/RangeUtils.h"
#include "vendor/distance_transform/include/distance_transform/distance_transform.hpp"

#include <stdio.h>      /* printf */
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>    // std::min
#include <time.h>
#include <chrono>
#include <set>
#include <iomanip>      // std::setw
#include <unistd.h>
#include <sstream>
// #define NDEBUG
#include <cassert>
#include <tuple>

#define _MAKE_TRACE_MAP 0
#define _TRACK_LUT_SIZE 0
#define _TRACK_COLLISION_INDEXES 0

#define _EPSILON 0.00001
#define M_2PI 6.28318530718
#define _BINARY_SEARCH_THRESHOLD 64 // if there are more than this number of elements in the lut bin, use binary search

// fast optimized version
#define _USE_CACHED_TRIG 0
#define _USE_ALTERNATE_MOD 1
#define _USE_CACHED_CONSTANTS 1
#define _USE_FAST_ROUND 0
#define _NO_INLINE 0
#define _USE_LRU_CACHE 0
#define _LRU_CACHE_SIZE 1000000

// not implemented yet -> use 16 bit integers to store zero points
#define _CDDT_SHORT_DATATYPE 1

#define _GIANT_LUT_SHORT_DATATYPE 1

// slow unoptimized version 
// #define _USE_ALTERNATE_MOD 0
// #define _USE_CACHED_CONSTANTS 1
// #define _USE_FAST_ROUND 1
// #define _DO_MOD 0 // this might not be necessary (aka 1 & 0 might be equivalent), will evaluate later
// #define _NO_INLINE 0
// 
#if _USE_LRU_CACHE
#include "includes/lru_cache.h"
#endif

// No inline
#if _NO_INLINE == 1
#define ANIL __attribute__ ((noinline))
#else
#define ANIL 
#endif

// these defines are for yaml serialization

#define T1 "  "
#define T2 T1 T1
#define T3 T1 T1 T1
#define T4 T2 T2

#define J1 "  "
#define J2 J1 J1
#define J3 J1 J1 J1
#define J4 J2 J2

namespace ranges {
	struct OMap
	{
		bool has_error;
		unsigned width;  // x axis
		unsigned height; // y axis
		std::vector<std::vector<bool> > grid;
		std::vector<std::vector<float> > raw_grid;
		std::string fn; // filename
		#if _MAKE_TRACE_MAP == 1
		std::vector<std::vector<bool> > trace_grid;
		#endif

		OMap(int w, int h) : width(w), height(h), fn(""), has_error(false) {
			for (int i = 0; i < w; ++i) {
				std::vector<bool> y_axis;
				for (int q = 0; q < h; ++q) y_axis.push_back(false);
				grid.push_back(y_axis);
			}
			#if _MAKE_TRACE_MAP == 1
			for (int i = 0; i < w; ++i) {
				std::vector<bool> y_axis;
				for (int q = 0; q < h; ++q) y_axis.push_back(false);
				trace_grid.push_back(y_axis);
			}
			#endif
		}

		OMap(std::string filename) : OMap(filename, 128) {}
		OMap(std::string filename, float threshold) : fn(filename), has_error(false) {
			unsigned error;
			unsigned char* image;

			error = lodepng_decode32_file(&image, &width, &height, filename.c_str());
			if(error) {
				printf("ERROR %u: %s\n", error, lodepng_error_text(error));
				has_error = true;
				return;
			}

			for (int i = 0; i < width; ++i) {
				std::vector<bool> y_axis;
				for (int q = 0; q < height; ++q) y_axis.push_back(false);
				grid.push_back(y_axis);
			}

			for (int i = 0; i < width; ++i) {
				std::vector<float> y_axis;
				for (int q = 0; q < height; ++q) y_axis.push_back(0);
				raw_grid.push_back(y_axis);
			}

			#if _MAKE_TRACE_MAP == 1
			for (int i = 0; i < width; ++i) {
				std::vector<bool> y_axis;
				for (int q = 0; q < height; ++q) y_axis.push_back(false);
				trace_grid.push_back(y_axis);
			}
			#endif

			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					unsigned idx = 4 * y * width + 4 * x;
					int r = image[idx + 2];
					int g = image[idx + 1];
					int b = image[idx + 0];
					int gray = (int) utils::rgb2gray(r,g,b);

					// std::cout << gray << std::endl;

					if (gray < threshold) grid[x][y] = true;
					raw_grid[x][y] = gray;
				}
			}
		}

		bool get(int x, int y) { return grid[x][y]; }
		bool isOccupied(int x, int y) { 
			#if _MAKE_TRACE_MAP == 1
			trace_grid[x][y] = true;
			#endif
			return grid[x][y]; 
		}

		// query the grid without a trace
		bool isOccupiedNT(int x, int y) { return grid[x][y]; }

		#if _MAKE_TRACE_MAP == 1
		bool saveTrace(std::string filename) {
			std::vector<unsigned char> png;
			lodepng::State state; //optionally customize this one
			// char image = new char[width * height * 4] = 0;
			char image[width * height * 4];

			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					unsigned idx = 4 * y * width + 4 * x;

					// if (trace_grid[x][y]) {
					// 	image[idx + 0] = 255;
					// 	image[idx + 1] = 255;
					// 	image[idx + 2] = 255;
					// }
					
					image[idx + 2] = 255;
					image[idx + 1] = 255;
					image[idx + 0] = 255;

					if (trace_grid[x][y]) {
						image[idx + 0] = 0;
						image[idx + 1] = 0;
						image[idx + 2] = 200;
					}

					if (grid[x][y]) {
						image[idx + 0] = 255;
						image[idx + 1] = 0;
						image[idx + 2] = 0;
					}

					if (grid[x][y] && trace_grid[x][y]) {
						image[idx + 0] = 0;
						image[idx + 1] = 0;
						image[idx + 2] = 0;
					}
					image[idx + 3] = 255;
				}
			}
			unsigned error = lodepng::encode(png, reinterpret_cast<const unsigned char*> (image), width, height, state);
			if(!error) lodepng::save_file(png, filename);
			//if there's an error, display it
			if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
			return error;
		}
		#endif

		bool save(std::string filename) {
			std::vector<unsigned char> png;
			lodepng::State state; //optionally customize this one
			// char image = new char[width * height * 4] = 0;
			char image[width * height * 4];

			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					unsigned idx = 4 * y * width + 4 * x;
					
					image[idx + 2] = 255;
					image[idx + 1] = 255;
					image[idx + 0] = 255;
					image[idx + 3] = 255;

					if (grid[x][y]) {
						image[idx + 0] = 0;
						image[idx + 1] = 0;
						image[idx + 2] = 0;
					}
				}
			}
			unsigned error = lodepng::encode(png, reinterpret_cast<const unsigned char*> (image), width, height, state);
			if(!error) lodepng::save_file(png, filename);
			//if there's an error, display it
			if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
			return error;
		}

		OMap make_edge_map(bool count_corners) {
			OMap edge_map = OMap(width, height);
			int occupied = 0;
			for (int x = 0; x < width; ++x) {
				for (int y = 0; y < height; ++y) {
					if (!isOccupiedNT(x,y)) continue;

					std::vector<std::pair<int,int>> outline = utils::outline(x,y,count_corners);
					for (int i = 0; i < outline.size(); ++i) {
						int cx;
						int cy;
						std::tie(cx, cy) = outline[i];
						if (0 <= cx && 0 <= cy && cx < width && cy < height && !isOccupiedNT(cx,cy)) {
							edge_map.grid[x][y] = true;
							break;
						}
					}
				}
			}
			return edge_map;
		}

		bool error() {
			return has_error;
		}

		// returns memory usage in bytes
		int memory() {
			// std::cout << sizeof(bool) * width * height << std::endl;
			return sizeof(bool) * width * height;
		}
	};

	struct DistanceTransform
	{
		unsigned width;
		unsigned height;
		std::vector<std::vector<float> > grid;

		float get(int x, int y) { return grid[x][y]; }

		DistanceTransform() : width(0), height(0) {}

		DistanceTransform(int w, int h) : width(w), height(h) {
			// allocate space in the vectors
			for (int i = 0; i < width; ++i) {
				std::vector<float> y_axis;
				for (int q = 0; q < height; ++q) y_axis.push_back(1.0);
				grid.push_back(y_axis);
			}
		}

		// computes the distance transform of a given OMap
		DistanceTransform(OMap *map) {
			width = map->width;
			height = map->height;

			dope::Index2 grid_size({width, height});
			dope::Grid<float, 2> f(grid_size);
			dope::Grid<dope::SizeType, 2> indices(grid_size);

			for (dope::SizeType i = 0; i < width; ++i)
		        for (dope::SizeType j = 0; j < height; ++j)
		        	if (map->isOccupied(i,j)) f[i][j] = 0.0;
		        	else f[i][j] = std::numeric_limits<float>::max();

			dt::DistanceTransform::initializeIndices(indices);
			dt::DistanceTransform::distanceTransformL2(f, f, indices, false, 1);


			// allocate space in the vectors
			for (int i = 0; i < width; ++i) {
				std::vector<float> y_axis;
				for (int q = 0; q < height; ++q) y_axis.push_back(1.0);
				grid.push_back(y_axis);
			}

			// store to array
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					grid[x][y] = f[x][y];
				}
			}
		}

		bool save(std::string filename) {
			std::vector<unsigned char> png;
			lodepng::State state; //optionally customize this one
			char image[width * height * 4];

			float scale = 0;
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					scale = std::max(grid[x][y], scale);
				}
			}
			scale *= 1.0 / 255.0;
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					unsigned idx = 4 * y * width + 4 * x;
					// std::cout << (int)(grid[x][y] / scale) << " " << grid[x][y] / scale << std::endl;
					// image[idx + 2] = std::min(255, (int)grid[x][y]);
					// image[idx + 1] = std::min(255, (int)grid[x][y]);
					// image[idx + 0] = std::min(255, (int)grid[x][y]);
					image[idx + 2] = (int)(grid[x][y] / scale);
					image[idx + 1] = (int)(grid[x][y] / scale);
					image[idx + 0] = (int)(grid[x][y] / scale);
					image[idx + 3] = 255;
				}
			}
			unsigned error = lodepng::encode(png, reinterpret_cast<const unsigned char*> (image), width, height, state);
			if(!error) lodepng::save_file(png, filename);
			//if there's an error, display it
			if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
			return error;
		}

		int memory() {
			return width*height*sizeof(float);
		}
	};

	class RangeMethod
	{
	public:
		RangeMethod(OMap m, float mr) : map(m), max_range(mr) {};
		virtual ~RangeMethod() {};

		virtual float calc_range(float x, float y, float heading) = 0;
		virtual std::pair<float,float> calc_range_pair(float x, float y, float heading) { return std::make_pair(-1,-1); }
		virtual OMap *getMap() {return &map; }
		virtual void report() {};
		float maxRange() { return max_range; }
		float memory() { return -1; }
		
		#if _MAKE_TRACE_MAP == 1
		void saveTrace(std::string fn) { map.saveTrace(fn); }
		#endif

		// wrapper function to call calc_range repeatedly with the given array of inputs
		// and store the result to the given outputs. Useful for avoiding cython function
		// call overhead by passing it a numpy array pointer. Indexing assumes a 3xn numpy array
		// for the inputs and a 1xn numpy array of the outputs
		void numpy_calc_range(float * ins, float * outs, int num_casts) {
			for (int i = 0; i < num_casts; ++i) {
				outs[i] = calc_range(ins[i], ins[i+num_casts], ins[i+num_casts*2]);
			}
		}
	
	protected:
		OMap map;
		float max_range;
	};

	class BresenhamsLine : public RangeMethod
	{
	public:
		BresenhamsLine(OMap m, float mr) : RangeMethod(m, mr) {};
		
		float ANIL calc_range(float x, float y, float heading) {
			// first check if the cell underneath the query point is occupied, if so return
			if (map.isOccupied((int)x,(int)y)) {
				return 0.0;
			}

			/*
			 this defines the coordinate system such that 
			    ------> +x
			    |
			    |
			    \/
			    +y
			  0* heading lies along the x axis, positive heading rotates towards the positive y axis
			*/
			float x0 = y;
			float y0 = x;
			float x1 = y + max_range*sinf(heading);
			float y1 = x + max_range*cosf(heading);

			bool steep = false;
			if (std::abs(y1-y0) > std::abs(x1-x0)) steep = true;

			if (steep) {
				float tmp = x0;
				x0 = y0;
				y0 = tmp;
				tmp = x1;
				x1 = y1;
				y1 = tmp;
			}

			float deltax = std::abs(x1-x0);
			float deltay = std::abs(y1-y0);

			float error = 0;
			float deltaerr = deltay;
			float _x = x0;
			float _y = y0;

			int xstep = -1;
			if (x0 < x1) xstep = 1;

			int ystep = -1;
			if (y0 < y1) ystep = 1;

			unsigned width = map.width;
			unsigned height = map.height;

			while ((int)_x != (int)(x1 + xstep)) {
				_x += xstep;
				error += deltaerr;

				if (error * 2.00 >= deltax) {
					_y += ystep;
					error -= deltax;
				}

				if (!steep) {
					if (0 <= _y && _y < width && 0 <= _x && _x < height && map.isOccupied(_y, _x)) {
						float xd = _x - x0;
						float yd = _y - y0;
						return sqrtf(xd*xd + yd*yd);
					}
				} else {
					if (0 <= _x && _x < width && 0 <= _y && _y < height && map.isOccupied(_x, _y)) {
						float xd = _x - x0;
						float yd = _y - y0;
						return sqrtf(xd*xd + yd*yd);
					}
				}
			}
			return max_range; 
		}

		int memory() { return map.memory(); }
	};

	class RayMarching : public RangeMethod
	{
	public:
		RayMarching(OMap m, float mr) : RangeMethod(m, mr) { distImage = DistanceTransform(&m); }
		
		float ANIL calc_range(float x, float y, float heading) {
			float x0 = x;
			float y0 = y;

			float ray_direction_x = cosf(heading);
			float ray_direction_y = sinf(heading);

			int px = 0;
			int py = 0;

			float t = 0.0;
			while (t < max_range) {
				px = x0 + ray_direction_x * t;
				py = y0 + ray_direction_y * t;

				if (px >= map.width || px < 0 || py < 0 || py >= map.height) {
					return max_range;
				}

				float d = distImage.get(px, py);
				
				#if _MAKE_TRACE_MAP == 1
				map.isOccupied(px,py); // this makes a dot appear in the trace map
				#endif

				if (d <= distThreshold) {
					float xd = px - x0;
					float yd = py - y0;
					return sqrtf(xd*xd + yd*yd);
				}

				t += std::max<float>(d * step_coeff, 1.0);
			}

			return max_range; 
		}

		int memory() { return distImage.memory(); }
	protected:
		DistanceTransform distImage;
		float distThreshold = 0.0;
		float step_coeff = 0.999;
	};

	class CDDTCast : public RangeMethod
	{
	public:
		CDDTCast(OMap m, float mr, unsigned int td) :  RangeMethod(m, mr), theta_discretization(td) { 
			#if _USE_CACHED_CONSTANTS
			theta_discretization_div_M_2PI = theta_discretization / M_2PI;
			M_2PI_div_theta_discretization = M_2PI / ((float) theta_discretization);
			#endif

			#if _USE_LRU_CACHE
			cache = cache::lru_cache<uint64_t,float>(_LRU_CACHE_SIZE, -1);
			key_maker = utils::KeyMaker<uint64_t>(m.width,m.height,theta_discretization);
			#endif


			// determines the width of the projection of the map along each angle
			std::vector<int> lut_widths;
			// the angle for each theta discretization bin
			std::vector<float> angles;

			// compute useful constants and cache for later use
			for (int i = 0; i < theta_discretization; ++i)
			{
				#if _USE_CACHED_CONSTANTS
				float angle = i * M_2PI_div_theta_discretization;
				#else
				float angle = M_2PI * i / theta_discretization;
				#endif
				angles.push_back(angle);

				#if _USE_CACHED_TRIG == 1
				float cosfangle = cosf(angle);
				float sinfangle = sinf(angle);
				cos_values.push_back(cosfangle);
				sin_values.push_back(sinfangle);
				#endif

				// compute the height of the axis aligned bounding box, which will determine
				// the necessary width of the lookup table for this angle
				#if _USE_CACHED_TRIG == 1
				float rotated_height = std::abs(map.width * sinfangle) + std::abs(map.height * cosfangle);
				#else
				float rotated_height = std::abs(map.width * sinf(angle)) + std::abs(map.height * cosf(angle));
				#endif
				unsigned int lut_width = ceil(rotated_height - _EPSILON);
				lut_widths.push_back(lut_width);

				/* the entire map will be rotated by the given angle. Every pixel in t hat map must be
				   projected into the LUT, so we need to make sure that the index of every pixel will be 
				   positive when projected into LUT space. For example, here's the example with no rotation

            (0,height)  (width,height)      {
						    *----------*    -----------> []
						    |  a       |    -----------> [a] 
						    |      b   |    -----------> [b] 
						    |  c      d|    -----------> [c,d]
						    *----------o    -----------> []
						  (0,0)       (width,0)           }

					 This is the case when theta = pi / 2 radians (not totally to scale)

				   (-height,width) (0,width)      {
					  	     *--------*    -----------> []
					  	     |       d|    -----------> [d] 
					  	     |   b    |    -----------> [b] 
					  	     |        |    -----------> []
					  	     | a   c  |    -----------> [a,c]
					  	     *--------o    -----------> []
				     (-height,0)  (0,0)                }

				   Notably, the corner labeled 'o' lies on the origin no matter the rotation. Therefore,
				   to ensure every LUT index is positive, we should translate the rotated map by:
				   	         max(0, -1 * [minimum y coordinate for each corner])
				*/
				
				// this is the y-coordinate for each non-origin corner
				#if _USE_CACHED_TRIG == 1
				float left_top_corner_y     = map.height * cosfangle;
				float right_top_corner_y    = map.width * sinfangle + map.height * cosfangle;
				float right_bottom_corner_y = map.width * sinfangle;
				#else
				float left_top_corner_y     = map.height * cosf(angle);
				float right_top_corner_y    = map.width * sinf(angle) + map.height * cosf(angle);
				float right_bottom_corner_y = map.width * sinf(angle);
				#endif
				

				// find the lowest corner, and determine the translation necessary to make them all positive
				float min_corner_y = std::min(left_top_corner_y, std::min(right_top_corner_y, right_bottom_corner_y));
				float lut_translation = std::max(0.0, -1.0 * min_corner_y - _EPSILON);

				lut_translations.push_back(lut_translation);
			}

			// build the empty LUT datastructure
			for (int a = 0; a < theta_discretization; ++a)
			{
				std::vector<std::vector<float> > projection_lut;
				for (int i = 0; i < lut_widths[a]; ++i)
				// for (int i = 0; i < 10; ++i)
				{
					std::vector<float> column;
					projection_lut.push_back(column);
				}
				compressed_lut.push_back(projection_lut);
			}

			// compute the edge map of the geometry - no ray can intersect with non-edge geometry,
			// so pruning it now will speed up LUT construction, especially for dense maps
			OMap edge_map = map.make_edge_map(false);
			// edge_map.save("./edge_map.png");

			// fill the LUT datastructure by projecting each occupied pixel into LUT space and storing
			// the x position in LUT space at the correct place as determined by y position and theta
			for (int x = 0; x < map.width; ++x) {
				for (int y = 0; y < map.height; ++y) {
					// if (map.isOccupied(x,y)) {
						if (edge_map.isOccupied(x,y)) {
						// this (x,y) is occupied, so add it to the datastruture
						std::pair<float, float> pixel_center =  std::make_pair(x + 0.5, y + 0.5);
						for (int a = 0; a < theta_discretization / 2.0; ++a) {
							#if _USE_CACHED_TRIG == 1
							float cosangle = cos_values[a];
							float sinangle = sin_values[a];
							#else
							float angle = angles[a];
							float cosangle = cosf(angle);
							float sinangle = sinf(angle);
							#endif

							float half_lut_space_width = (std::abs(sinangle) + std::abs(cosangle)) / 2.0;

							float lut_space_center_x = pixel_center.first * cosangle - pixel_center.second * sinangle;
							float lut_space_center_y = (pixel_center.first * sinangle + pixel_center.second * cosangle) + lut_translations[a];

							int upper_bin = lut_space_center_y + half_lut_space_width - _EPSILON;
							int lower_bin = lut_space_center_y - half_lut_space_width + _EPSILON;

							for (int i = lower_bin; i <= upper_bin; ++i) 
								compressed_lut[a][i].push_back(lut_space_center_x);

							// std::cout << std::endl;
							// std::cout << "angle: " << angle << std::endl;
							// std::cout << "center: (" << pixel_center.first << ", " << pixel_center.second << ")" << std::endl;
							// std::cout << "new center: (" << lut_space_center_x << ", " << lut_space_center_y << ")" << std::endl;
							// std::cout << "bins:" << upper_bin << "    " << (int) lut_space_center_y << "   " << lower_bin << std::endl;
							// std::cout << "width:" << half_lut_space_width << std::endl;
							// std::cout << "trans" << lut_translations[a] << std::endl;
							// std::cout << upper_bin << "   " << lower_bin << "   " << lut_translations[a] << std::endl;
							// std::cout << lut_space_center_x << "  " << lut_space_center_y << std::endl;
						}
					}
				}
			}

			// sort the vectors for faster lookup with binary search
			for (int a = 0; a < theta_discretization; ++a)
			{
				for (int i = 0; i < compressed_lut[a].size(); ++i)
				{
					// sort the vectors
					std::sort(compressed_lut[a][i].begin(), compressed_lut[a][i].end());

					// remove all duplicate entries, they will not change the answer
					compressed_lut[a][i].erase( unique( compressed_lut[a][i].begin(), compressed_lut[a][i].end() ), compressed_lut[a][i].end());
				}
			}

			#if _TRACK_LUT_SIZE == 1
				std::cout << "LUT SIZE (MB): " << lut_size() / 1000000.0 << std::endl;
			#endif

			#if _TRACK_COLLISION_INDEXES == 1
			for (int a = 0; a < theta_discretization; ++a) {
				std::vector<std::set<int> > projection_lut_tracker;
				for (int i = 0; i < lut_widths[a]; ++i)
				// for (int i = 0; i < 10; ++i)
				{
					std::set<int> collection;
					projection_lut_tracker.push_back(collection);
				}
				collision_table.push_back(projection_lut_tracker);
			}
			#endif
		}

		int lut_size() {
			int lut_size = 0;
			// sort the vectors for faster lookup with binary search
			for (int a = 0; a < theta_discretization; ++a) {
				for (int i = 0; i < compressed_lut[a].size(); ++i) {
					lut_size += compressed_lut[a][i].size();
				}
			}
			return lut_size * sizeof(float);
		}

		int memory() { return lut_size()+map.memory()+lut_translations.size()*sizeof(float); }

		// mark all of the LUT entries that are potentially useful
		// remove all LUT entries that are not potentially useful
		void prune(float max_range) {

			std::vector<std::vector<std::set<int> > > local_collision_table;

			for (int a = 0; a < theta_discretization / 2.0; ++a) {
				std::vector<std::set<int> > projection_lut_tracker;
				for (int i = 0; i < compressed_lut[a].size(); ++i) {
					std::set<int> collection;
					projection_lut_tracker.push_back(collection);
				}
				local_collision_table.push_back(projection_lut_tracker);
			}

			for (int angle_index = 0; angle_index < theta_discretization / 2.0; ++angle_index) {
				#if _USE_CACHED_CONSTANTS
				float angle = angle_index * M_2PI_div_theta_discretization;
				#else
				float angle = M_2PI * angle_index / theta_discretization;
				#endif

				#if _USE_CACHED_TRIG == 1
				float cosangle = cos_values[angle_index];
				float sinangle = sin_values[angle_index];
				#else
				float cosangle = cosf(angle);
				float sinangle = sinf(angle);
				#endif

				// float cosangle = cos_values[angle_index];
				// float sinangle = sin_values[angle_index];
				float translation = lut_translations[angle_index];
				
				float lut_space_x;
				float lut_space_y;
				unsigned int lut_index;
				std::vector<float> *lut_bin;
				for (int x = 0; x < map.grid.size(); ++x) {
					float _x = 0.5 + x;
					for (int y = 0; y < map.grid[0].size(); ++y) {
						float _y = 0.5 + y;
						lut_space_x = _x * cosangle - _y * sinangle;
						lut_space_y = (_x * sinangle + _y * cosangle) + translation;
						lut_index = (int) lut_space_y;

						lut_bin = &compressed_lut[angle_index][lut_index];

						// binary search for next greatest element
						int low = 0;
						int high = lut_bin->size() - 1;

						// there are no entries in this lut bin
						if (high == -1) continue;
						if (map.grid[x][y]) continue;

						// the furthest entry is behind the query point
						// if ((*lut_bin)[high] + max_range < lut_space_x) return std::make_pair(max_range, max_range);
						if ((*lut_bin)[high] < lut_space_x && lut_space_x - (*lut_bin)[high] < max_range) {
							local_collision_table[angle_index][lut_index].insert(high);
							// accum += 1;
							continue;
						}

						int index;
						if (high > _BINARY_SEARCH_THRESHOLD) {
							// once the binary search terminates, the next greatest element is indicated by 'val'
							index = std::lower_bound(lut_bin->begin(), lut_bin->end(), lut_space_x) - lut_bin->begin();
						} else { // do linear search if array is very small
							for (int i = 0; i < lut_bin->size(); ++i) {
								if ((*lut_bin)[i] >= lut_space_x) {
									index = i;
									break;
								}
							}
						}

						int inverse_index = index - 1;
						if (inverse_index == -1) {
							local_collision_table[angle_index][lut_index].insert(index);
							continue;
						} else {
							local_collision_table[angle_index][lut_index].insert(index);
							local_collision_table[angle_index][lut_index].insert(inverse_index);
							continue;
						}

					}
				}
			}

			#if _TRACK_LUT_SIZE == 1
			std::cout << "OLD LUT SIZE (MB): " << lut_size() / 1000000.0 << std::endl;
			#endif

			for (int a = 0; a < theta_discretization / 2.0; ++a) {
				for (int lut_index = 0; lut_index < compressed_lut[a].size(); ++lut_index) {
					std::vector<float> pruned_bin;

					for (int i = 0; i < compressed_lut[a][lut_index].size(); ++i) {
						bool is_used = local_collision_table[a][lut_index].find(i) != local_collision_table[a][lut_index].end();
						if (is_used) pruned_bin.push_back(compressed_lut[a][lut_index][i]);
					}
					compressed_lut[a][lut_index] = pruned_bin;
				}
			}

			#if _TRACK_LUT_SIZE == 1
			std::cout << "NEW LUT SIZE (MB): " << lut_size() / 1000000.0 << std::endl;
			#endif
		}

		// takes a continuous theta space and returns the nearest theta in the discrete LUT space
		// as well as the bin index that the given theta falls into
		std::tuple<int, float, bool>  discretize_theta(float theta) {
			#if _USE_ALTERNATE_MOD
			if (theta < 0.0) {
				while (theta < 0.0) {
					theta += M_2PI;
				}
			} else if (theta > M_2PI) {
				while (theta > M_2PI) {
					theta -= M_2PI;
				}
			}
			#else
			theta = fmod(theta, M_2PI);
			// fmod does not wrap the angle into the positive range, so this will fix that if necessary
			if (theta < 0.0)
   			theta += M_2PI;
   			#endif

			// exploit rotational symmetry by wrapping the theta range around to the range 0:pi
			bool is_flipped = false;
			if (theta >= M_PI) {
				is_flipped = true;
				theta -= M_PI;
			}

   			#if _USE_CACHED_CONSTANTS == 1
	   		#if _USE_FAST_ROUND == 1
	   		int rounded = int (theta * theta_discretization_div_M_2PI + 0.5);
	   		#else
	   		int rounded = (int) roundf(theta * theta_discretization_div_M_2PI);
	   		#endif
	   		int binned = rounded % theta_discretization;
				float discrete_angle = binned * M_2PI_div_theta_discretization;
   			#else
	   		#if _USE_FAST_ROUND == 1
	   		int rounded = int ((theta * theta_discretization / M_2PI) + 0.5);
	   		#else
	   		int rounded = (int) roundf(theta * theta_discretization / M_2PI);
	   		#endif
	   		int binned = rounded % theta_discretization;
				float discrete_angle = (binned * M_2PI) / ((float) theta_discretization);
			#endif
			return std::make_tuple(binned, discrete_angle, is_flipped);
		}

		float ANIL calc_range(float x, float y, float heading) {
			
			#if _USE_LRU_CACHE
			int theta_key = (int) roundf(heading * theta_discretization_div_M_2PI);
			// int theta_key = angle_index;
			// if (is_flipped)
			// 	theta_key += theta_discretization/2;
			uint64_t key = key_maker.make_key(int(x), int(y), theta_key);
			float val = cache.get(key);
			if (val > 0) {
				hits += 1;
				return val;
			} else {
				misses += 1;
			}
			// if (cache.exists(key)) { return cache.get(key); }
			#endif

			int angle_index;
			float discrete_theta;
			bool is_flipped;
			std::tie(angle_index, discrete_theta, is_flipped) = discretize_theta(-1.0*heading);

			#if _USE_CACHED_TRIG == 1
			float cosangle = cos_values[angle_index];
			float sinangle = sin_values[angle_index];
			#else
			float cosangle = cosf(discrete_theta);
			float sinangle = sinf(discrete_theta);
			#endif

			float lut_space_x = x * cosangle - y * sinangle;
			float lut_space_y = (x * sinangle + y * cosangle) + lut_translations[angle_index];

			unsigned int lut_index = (int) lut_space_y;
			std::vector<float> *lut_bin = &compressed_lut[angle_index][lut_index];

			// the angle is in range pi:2pi, so we must search in the opposite direction
			if (is_flipped) {
				// std::cout << "flipped" << std::endl;
				// binary search for next greatest element
				int low = 0;
				int high = lut_bin->size() - 1;

				// there are no entries in this lut bin
				if (high == -1) {
					#if _USE_LRU_CACHE
					cache.put(key, max_range);
					#endif
					return max_range;
				}
				// the furthest entry is behind the query point
				if ((*lut_bin)[low] > lut_space_x) {
					#if _USE_LRU_CACHE
					cache.put(key, max_range);
					#endif
					return max_range;
				}
				if ((*lut_bin)[high]< lut_space_x) {
					float val = lut_space_x - (*lut_bin)[high];
					#if _USE_LRU_CACHE
					cache.put(key, val);
					#endif
					return val;
				}

				// the query point is on top of a occupied pixel
				// this call is here rather than at the beginning, because it is apparently more efficient.
				// I presume that this has to do with the previous two return statements
				if (map.grid[x][y]) { return 0.0; }

				if (high > _BINARY_SEARCH_THRESHOLD) {
					int index = std::upper_bound(lut_bin->begin(), lut_bin->end(), lut_space_x) - lut_bin->begin();
					assert(index > 0); // if index is 0, this will segfault. that should never happen, though.
					float val = lut_space_x - (*lut_bin)[index-1];
					
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					#endif

					#if _USE_LRU_CACHE
					cache.put(key, val);
					#endif
					return val;
				} else { // do linear search if array is very small
					for (int i = high; i >= 0; --i)
					{
						float obstacle_x = (*lut_bin)[i];
						if (obstacle_x <= lut_space_x) {
							#if _TRACK_COLLISION_INDEXES == 1
							collision_table[angle_index][lut_index].insert(i);
							#endif

							float val = lut_space_x - obstacle_x;
							#if _USE_LRU_CACHE
							cache.put(key, val);
							#endif
							return val;
						}
					}
				}
			} else {
				// std::cout << "not flipped" << std::endl;
				// binary search for next greatest element
				int low = 0;
				int high = lut_bin->size() - 1;

				// there are no entries in this lut bin
				if (high == -1) {
					#if _USE_LRU_CACHE
					cache.put(key, max_range);
					#endif
					return max_range;
				}
				// the furthest entry is behind the query point
				if ((*lut_bin)[high] < lut_space_x) {
					#if _USE_LRU_CACHE
					cache.put(key, max_range);
					#endif
					return max_range;
				}
				if ((*lut_bin)[low] > lut_space_x) {
					float val = (*lut_bin)[low] - lut_space_x;
					#if _USE_LRU_CACHE
					cache.put(key, val);
					#endif
					return val;
				}
				// the query point is on top of a occupied pixel
				// this call is here rather than at the beginning, because it is apparently more efficient.
				// I presume that this has to do with the previous two return statements
				if (map.grid[x][y]) { return 0.0; }

				if (high > _BINARY_SEARCH_THRESHOLD) {
					// once the binary search terminates, the next greatest element is indicated by 'val'
					// float val = *std::lower_bound(lut_bin->begin(), lut_bin->end(), lut_space_x);
					int index = std::upper_bound(lut_bin->begin(), lut_bin->end(), lut_space_x) - lut_bin->begin();
					float val = (*lut_bin)[index] - lut_space_x;
					
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					#endif

					#if _USE_LRU_CACHE
					cache.put(key, val);
					#endif

					return val;
				} else { // do linear search if array is very small
					// std::cout << "L" ;//<< std::endl;
					for (int i = 0; i < lut_bin->size(); ++i)
					{
						float obstacle_x = (*lut_bin)[i];
						if (obstacle_x >= lut_space_x) {
							#if _TRACK_COLLISION_INDEXES == 1
							collision_table[angle_index][lut_index].insert(i);
							#endif

							float val = obstacle_x - lut_space_x;

							#if _USE_LRU_CACHE
							cache.put(key, val);
							#endif

							return val;
						}
					}
				}
			}
			// this should never occur, if it does, there's an error
			assert(0);
			return -1.0;
		}

		// returns both range for the given heading, and heading + pi/2
		// it is efficient to do both at the same time, rather than both
		// independently if they are both required
		std::pair<float,float> calc_range_pair(float x, float y, float heading) {
			int angle_index;
			float discrete_theta;
			bool is_flipped;
			std::tie(angle_index, discrete_theta, is_flipped) = discretize_theta(-1.0*heading);

			#if _USE_CACHED_TRIG == 1
			float cosangle = cos_values[angle_index];
			float sinangle = sin_values[angle_index];
			#else
			float cosangle = cosf(discrete_theta);
			float sinangle = sinf(discrete_theta);
			#endif

			float lut_space_x = x * cosangle - y * sinangle;
			float lut_space_y = (x * sinangle + y * cosangle) + lut_translations[angle_index];

			unsigned int lut_index = (int) lut_space_y;
			std::vector<float> *lut_bin = &compressed_lut[angle_index][lut_index];

			// the angle is in range pi:2pi, so we must search in the opposite direction
			if (is_flipped) {
				// std::cout << "is flipped" << std::endl;
				// binary search for next greatest element
				int low = 0;
				int high = lut_bin->size() - 1;

				// there are no entries in this lut bin
				if (high == -1) return std::make_pair(max_range, max_range);
				// the furthest entry is behind the query point and out of max range of the inverse query
				// if ((*lut_bin)[low] - max_range > lut_space_x) return std::make_pair(max_range, max_range);				
				if ((*lut_bin)[low] > lut_space_x) 
					return std::make_pair(max_range, std::min(max_range, (*lut_bin)[low] - lut_space_x));
				if ((*lut_bin)[high]< lut_space_x) 
					return std::make_pair(lut_space_x - (*lut_bin)[high], max_range);
				// the query point is on top of a occupied pixel
				// this call is here rather than at the beginning, because it is apparently more efficient.
				// I presume that this has to do with the previous two return statements
				if (map.grid[x][y]) { return std::make_pair(0.0,0.0); }

				float val;
				int index;
				if (high > _BINARY_SEARCH_THRESHOLD) {
					// once the binary search terminates, the next least element is indicated by 'val'
					// float val = *std::lower_bound(lut_bin->begin(), lut_bin->end(), lut_space_x);
					index = std::upper_bound(lut_bin->begin(), lut_bin->end(), lut_space_x) - lut_bin->begin() - 1;
					val = (*lut_bin)[index];
				} else { // do linear search if array is very small
					for (int i = high; i >= 0; --i) {
						float obstacle_x = (*lut_bin)[i];
						if (obstacle_x <= lut_space_x) {
							index = i;
							val = obstacle_x;
							break;
						}
					}
				}

				// return std::make_pair(lut_space_x - val, max_range);

				// std::cout << (*lut_bin)[inverse_index+1] << std::endl;

				int inverse_index = index+1;
				if (inverse_index == lut_bin->size()) {
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					#endif

					return std::make_pair(lut_space_x - val, max_range);
				} else {
					
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					collision_table[angle_index][lut_index].insert(inverse_index);
					#endif

					return std::make_pair(lut_space_x - val, (*lut_bin)[inverse_index] - lut_space_x);
				}
			} else {
				// std::cout << "flipped" << std::endl;
				// binary search for next greatest element
				int low = 0;
				int high = lut_bin->size() - 1;

				// there are no entries in this lut bin
				if (high == -1) return std::make_pair(max_range, max_range);
				// the furthest entry is behind the query point
				// if ((*lut_bin)[high] + max_range < lut_space_x) return std::make_pair(max_range, max_range);
				if ((*lut_bin)[high] < lut_space_x) 
					return std::make_pair(max_range, std::min(max_range, lut_space_x - (*lut_bin)[high]));
				// TODO might need another early return case here
					// return std::make_pair(max_range, std::min(max_range, lut_space_x - (*lut_bin)[high]));
				// the query point is on top of a occupied pixel
				// this call is here rather than at the beginning, because it is apparently more efficient.
				// I presume that this has to do with the previous two return statements
				if (map.grid[x][y]) { std::make_pair(0.0, 0.0); }

				float val;
				int index;
				if (high > _BINARY_SEARCH_THRESHOLD) {
					// once the binary search terminates, the next greatest element is indicated by 'val'
					// float val = *std::lower_bound(lut_bin->begin(), lut_bin->end(), lut_space_x);
					index = std::lower_bound(lut_bin->begin(), lut_bin->end(), lut_space_x) - lut_bin->begin();
					val = (*lut_bin)[index];
					
					
				} else { // do linear search if array is very small
					// std::cout << "L" ;//<< std::endl;
					for (int i = 0; i < lut_bin->size(); ++i)
					{
						float obstacle_x = (*lut_bin)[i];
						if (obstacle_x >= lut_space_x) {
							val = obstacle_x;
							index = i;
							break;
							// return obstacle_x - lut_space_x;
						}
					}
				}


				int inverse_index = index - 1;
				if (inverse_index == -1) {
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					#endif

					return std::make_pair(val - lut_space_x, max_range);
				} else {
					
					#if _TRACK_COLLISION_INDEXES == 1
					collision_table[angle_index][lut_index].insert(index);
					collision_table[angle_index][lut_index].insert(inverse_index);
					#endif

					return std::make_pair(val - lut_space_x, lut_space_x - (*lut_bin)[inverse_index]);
				}
			}
		}

		void serializeYaml(std::stringstream* ss) {
			// (*ss) << std::fixed;
    		(*ss) << std::setprecision(7);

			(*ss) << "cddt:" << std::endl;
			(*ss) << T1 << "theta_discretization: " << theta_discretization << std::endl;
			(*ss) << T1 << "lut_translations: ";
			utils::serialize(lut_translations, ss);
			(*ss) << std::endl;
			(*ss) << T1 << "max_range: " << max_range << std::endl;
			(*ss) << T1 << "map: " << std::endl;
			(*ss) << T2 << "# note: map data is width and then height (width is number of rows) transposed from expectation:"  << std::endl;
			(*ss) << T2 << "path: " << map.fn << std::endl;
			(*ss) << T2 << "width: " << map.width << std::endl;
			(*ss) << T2 << "height: " << map.height << std::endl;
			(*ss) << T2 << "data: " << std::endl;
			for (int i = 0; i < map.width; ++i) {
				(*ss) << T3 << "- "; utils::serialize(map.grid[i], ss);(*ss) << std::endl;
			}
			(*ss) << T1 << "compressed_lut: " << std::endl;
			for (int i = 0; i < compressed_lut.size(); ++i) {
				#if _USE_CACHED_CONSTANTS
				float angle = i * M_2PI_div_theta_discretization;
				#else
				float angle = M_2PI * i / theta_discretization;
				#endif

				(*ss) << T2 << "- slice: " << std::endl;
				(*ss) << T3 << "theta: " << angle << std::endl;
				(*ss) << T3 << "zeros: " << std::endl;

				for (int j = 0; j < compressed_lut[i].size(); ++j) {
					(*ss) << T4 << "- "; utils::serialize(compressed_lut[i][j], ss); (*ss) << std::endl;
				}		
			}
		}

		void serializeJson(std::stringstream* ss) {
			// (*ss) << std::fixed;
    		(*ss) << std::setprecision(7);

			(*ss) << "{\"cddt\": {" << std::endl;
				(*ss)  << J1 << "\"theta_discretization\": " << theta_discretization  << ","<< std::endl;
				(*ss)  << J1 << "\"lut_translations\": "; utils::serialize(lut_translations, ss); (*ss) << "," << std::endl;
				(*ss) << J1 << "\"max_range\":" << max_range << "," << std::endl;
				(*ss) << J1 << "\"map\": {" << std::endl;
				// (*ss) << J2 << "# note: map data is width and then height (width is number of rows) transposed from expectation:"  << std::endl;
				(*ss) << J2 << "\"path\": \"" << map.fn << "\"," << std::endl;
				(*ss) << J2 << "\"width\": " << map.width << "," << std::endl;
				(*ss) << J2 << "\"height\": " << map.height << "," << std::endl;
				
				(*ss) << J2 << "\"data\": [";// utils::serialize(map.grid[0], ss);
				for (int i = 0; i < map.width; ++i) {
					if (i > 0) (*ss) << ","; 
					utils::serialize(map.grid[i], ss);
				}
				(*ss) << "]," << std::endl;
				(*ss) << J1 << "}," << std::endl;
				(*ss) << J1 << "\"compressed_lut\": [" << std::endl;
				for (int i = 0; i < compressed_lut.size(); ++i) {
					#if _USE_CACHED_CONSTANTS
					float angle = i * M_2PI_div_theta_discretization;
					#else
					float angle = M_2PI * i / theta_discretization;
					#endif

					(*ss) << J2 << "{" << std::endl;
					(*ss) << J3 << "\"theta\": " << angle << "," << std::endl;
					(*ss) << J3 << "\"zeros\": [";

					for (int j = 0; j < compressed_lut[i].size(); ++j) {
						if (j > 0) (*ss) << ","; 
						utils::serialize(compressed_lut[i][j], ss);
					}
					(*ss) << "]" << std::endl;
					if (i == compressed_lut.size() -1)	
						(*ss) << J2 << "}" << std::endl;
					else
						(*ss) << J2 << "}," << std::endl;
				}
				(*ss) << J1 << "]" << std::endl;
			(*ss) << "}}" << std::endl;
		}

		void report() {
			#if _USE_LRU_CACHE
			std::cout << "cache hits: " << hits << "  cache misses: " << misses << std::endl; 
			#endif
		}
	// protected:
		unsigned int theta_discretization;

		// compressed_lut[theta][offset] -> list of obstacle positions
		std::vector<std::vector<std::vector<float> > > compressed_lut;
		// std::vector<std::vector<bool > > map_grid;
		// cached list of y translations necessary to project points into lut space
		std::vector<float> lut_translations;
		

		#if _USE_CACHED_TRIG == 1
		std::vector<float> cos_values;
		std::vector<float> sin_values;
		#endif

		#if _USE_CACHED_CONSTANTS == 1
		float theta_discretization_div_M_2PI;
		float M_2PI_div_theta_discretization;
		#endif

		#if _TRACK_COLLISION_INDEXES == 1
		std::vector<std::vector<std::set<int> > > collision_table;
		#endif

		#if _USE_LRU_CACHE
		cache::lru_cache<uint64_t, float> cache;
		utils::KeyMaker<uint64_t> key_maker;
		int hits = 0;
		int misses = 0;
		#endif
	};

	class GiantLUTCast : public RangeMethod
	{
	public:
		#if _GIANT_LUT_SHORT_DATATYPE
		typedef uint16_t lut_t;
		#else
		typedef float lut_t;
		#endif

		GiantLUTCast(OMap m, float mr, int td) : theta_discretization(td), RangeMethod(m, mr) { 
			#if _USE_CACHED_CONSTANTS
			theta_discretization_div_M_2PI = theta_discretization / M_2PI;
			M_2PI_div_theta_discretization = M_2PI / ((float) theta_discretization);
			max_div_limits = max_range/std::numeric_limits<uint16_t>::max();
			limits_div_max = std::numeric_limits<uint16_t>::max() / max_range;
			#endif
			RayMarching seed_cast = RayMarching(m, mr);
			// CDDTCast seed_cast = CDDTCast(m, mr, td);

			for (int x = 0; x < m.width; ++x) {
				std::vector<std::vector<lut_t> > lut_slice;
				for (int y = 0; y < m.height; ++y) {
					std::vector<lut_t> lut_row;
					for (int i = 0; i < theta_discretization; ++i) {
						#if _USE_CACHED_CONSTANTS
						float angle = i * M_2PI_div_theta_discretization;
						#else
						float angle = M_2PI * i / theta_discretization;
						#endif
						float r = seed_cast.calc_range(x,y,angle);

						#if _GIANT_LUT_SHORT_DATATYPE
						r = std::min(max_range, r);
						#if _USE_CACHED_CONSTANTS
						uint16_t val = r * limits_div_max;
						#else
						uint16_t val = (r / max_range) * std::numeric_limits<uint16_t>::max();
						#endif
						lut_row.push_back(val);
						#else
						lut_row.push_back(r);
						#endif
					}
					lut_slice.push_back(lut_row);
				}
				giant_lut.push_back(lut_slice);
			}

			#if _TRACK_LUT_SIZE
			std::cout << "LUT SIZE (MB): " << lut_size() / 1000000.0 << std::endl;
			#endif
		}

		int lut_size() {
			return map.width * map.height * theta_discretization * sizeof(lut_t);
		}

		int memory() { return lut_size(); }

		// takes a continuous theta space and returns the nearest theta in the discrete LUT space
		// as well as the bin index that the given theta falls into
		int discretize_theta(float theta) {
			#if _USE_ALTERNATE_MOD
			if (theta < 0.0) {
				while (theta < 0.0) {
					theta += M_2PI;
				}
			} else if (theta > M_2PI) {
				while (theta > M_2PI) {
					theta -= M_2PI;
				}
			}
			#else
			theta = fmod(theta, M_2PI);
			// fmod does not wrap the angle into the positive range, so this will fix that if necessary
			if (theta < 0.0)
   			theta += M_2PI;
   			#endif

   			#if _USE_CACHED_CONSTANTS == 1
	   		#if _USE_FAST_ROUND == 1
	   		int rounded = int (theta * theta_discretization_div_M_2PI + 0.5);
	   		#else
	   		int rounded = (int) roundf(theta * theta_discretization_div_M_2PI);
	   		#endif
	   		int binned = rounded % theta_discretization;
   			#else
	   		#if _USE_FAST_ROUND == 1
	   		int rounded = int ((theta * theta_discretization / M_2PI) + 0.5);
	   		#else
	   		int rounded = (int) roundf(theta * theta_discretization / M_2PI);
	   		#endif
	   		int binned = rounded % theta_discretization;
			#endif
			return binned;
		}

		float ANIL calc_range(float x, float y, float heading) {
			#if _GIANT_LUT_SHORT_DATATYPE
				#if _USE_CACHED_CONSTANTS
			return giant_lut[(int)x][(int)y][discretize_theta(heading)] * max_div_limits;
				#else
			return max_range * giant_lut[(int)x][(int)y][discretize_theta(heading)] / std::numeric_limits<uint16_t>::max();
				#endif
			#else
			return giant_lut[(int)x][(int)y][discretize_theta(heading)];
			#endif
		}

		DistanceTransform *get_slice(float theta) {
			int width = giant_lut.size();
			int height = giant_lut[0].size();
			DistanceTransform *slice = new DistanceTransform(width, height);

			int dtheta = discretize_theta(theta);

			for (int x = 0; x < width; ++x) {
				for (int y = 0; y < height; ++y) {
					slice->grid[x][y] = giant_lut[x][y][dtheta];
					// slice->grid[x][y] = 100;
					// std::cout << giant_lut[x][y][dtheta] << std::endl;
				}
			}

			return slice;
		}
	protected:
		int theta_discretization;
		#if _USE_CACHED_CONSTANTS
		float theta_discretization_div_M_2PI;
		float M_2PI_div_theta_discretization;
		float max_div_limits;
		float limits_div_max;
		#endif
		std::vector<std::vector<std::vector<lut_t> > > giant_lut;
	};
} 

namespace benchmark {
	template <class range_T>
	class Benchmark
	{
	public:
		Benchmark(range_T rm) : range(rm) { 
			map = range.getMap();
		};
		~Benchmark() {};

		void set_log(std::stringstream* ss) { log = ss; }

		int memory() { return range.memory(); }

		void grid_sample(int step_size, int num_rays, int samples) {
			float coeff = (2.0 * M_PI) / num_rays;
			double t_accum = 0;
			float num_cast = 0;

			volatile clock_t t;
			t = clock();

			if (log) (*log) << "x,y,theta,time" << std::endl;
			if (log) (*log) << std::fixed;
    		if (log) (*log) << std::setprecision(9);

			for (int i = 0; i < num_rays; ++i)
			{
				float angle = i * coeff;
				for (int x = 0; x < map->width; x += step_size)
				{
					for (int y = 0; y < map->height; y += step_size)
					{
						auto start_time = std::chrono::high_resolution_clock::now();
						for (int i = 0; i < samples; ++i)
						{
							volatile float r = range.calc_range(x,y,angle);
						}

						auto end_time = std::chrono::high_resolution_clock::now();

						num_cast += samples;
						// std::cout << (end_time - start_time).count() << std::endl;

						std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

						t_accum += time_span.count();

						if (log) (*log) << x << "," << y << "," << angle << "," << time_span.count() << std::endl;
					}
				}
			}

			std::cout << "finished grid sample after: " << (((float) (clock() - t)) / CLOCKS_PER_SEC) << " sec" << std::endl;
			std::cout << " -avg time per ray: " << ( t_accum / (float) num_cast) << " sec" << std::endl;
			std::cout << " -rays cast: " << num_cast << std::endl;
			std::cout << " -total time: " << t_accum << " sec" << std::endl;
		}

		void random_sample(int num_samples) {
			std::default_random_engine generator;
			generator.seed(clock());
			std::uniform_real_distribution<float> randx = std::uniform_real_distribution<float>(1.0,map->width - 1.0);
			std::uniform_real_distribution<float> randy = std::uniform_real_distribution<float>(1.0,map->height - 1.0);
			std::uniform_real_distribution<float> randt = std::uniform_real_distribution<float>(0.0,M_2PI);

			double t_accum = 0;
			for (int i = 0; i < num_samples; ++i)
			{
				float x = randx(generator);
				float y = randy(generator);
				float angle = randt(generator);

				auto start_time = std::chrono::high_resolution_clock::now();
				volatile float r = range.calc_range(x,y,angle);
				auto end_time = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

				t_accum += time_span.count();
				if (log) (*log) << x << "," << y << "," << angle << "," << time_span.count() << std::endl;
			}

			std::cout << "finished random sample after: " << t_accum << " sec" << std::endl;
			std::cout << " -avg time per ray: " << ( t_accum / (float) num_samples) << " sec" << std::endl;
			std::cout << " -rays cast: " << num_samples << std::endl;
		}

		ranges::OMap *getMap() {return range.getMap(); }

	protected:
		range_T range;
		ranges::OMap *map;
		std::stringstream* log = NULL;
	};
}

#endif