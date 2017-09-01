#ifndef _RANGE_UTILS_H_INCLUDED_
#define	_RANGE_UTILS_H_INCLUDED_

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <climits>
#include <random>
#include <tuple>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <limits>
#include <set>
#include <chrono>
#include "btree_set.h"

#define M_2PI 6.28318530718

namespace utils {
	static unsigned long x=123456789, y=362436069, z=521288629;

	unsigned long xorshf96(void) {          //period 2^96-1
		// return std::rand() / RAND_MAX;
		unsigned long t;
		x ^= x << 16;
		x ^= x >> 5;
		x ^= x << 1;

		t = x;
		x = y;
		y = z;
		z = t ^ x ^ y;

		return z;
	}
	float rgb2gray(float r, float g, float b) {
		return 0.229 * r + 0.587 * g + 0.114 * b;
	}
	int randrange(int min, int max) {
		return min + (rand() % (int)(max - min + 1));
	}
	std::vector<std::pair<int,int> > outline(int x, int y, bool use_corners) {
		std::vector<std::pair<int,int> > corners;

		corners.push_back(std::make_pair(x+1,y));
		corners.push_back(std::make_pair(x-1,y));
		corners.push_back(std::make_pair(x,y+1));
		corners.push_back(std::make_pair(x,y-1));

		if (use_corners) {
			corners.push_back(std::make_pair(x+1,y+1));
			corners.push_back(std::make_pair(x-1,y+1));
			corners.push_back(std::make_pair(x+1,y-1));
			corners.push_back(std::make_pair(x-1,y-1));
		}

		return corners;
	}
	template <class key_T>
	class KeyMaker
	{
	public:
		KeyMaker() {}
		KeyMaker(int width, int height, int theta_discretization) {
			y_shift = (int) std::ceil(std::log2(theta_discretization));
			x_shift = (int) std::ceil(std::log2(height)) + y_shift;
			int bitness = (int) std::ceil(std::log2(width)) + x_shift;

			if (bitness > std::log2(std::numeric_limits<key_T>::max())) {
				std::cerr << "Key bitness too large for integer packing scheme. Check your KeyMaker template type." << std::endl;
			}

			// make bit masks for unpacking the various values
			t_mask = std::pow(2, y_shift)-1;
			y_mask = std::pow(2, x_shift)-1 - t_mask;
			x_mask = std::pow(2, bitness)-1 - y_mask - t_mask;
		};
		~KeyMaker() {};
		key_T make_key(int x, int y, int t) {
			return ((key_T)x << x_shift) + ((key_T)y << y_shift) + (key_T)t;
		}
		std::tuple<int, int, int> unpack_key(key_T k) {
			return std::make_tuple((int)((k & x_mask) >> x_shift), (int)((k & y_mask) >> y_shift), k & t_mask);
		}
	private:
		int y_shift;
		int x_shift;
		key_T x_mask;
		key_T y_mask;
		key_T t_mask;
	};

	enum Method {
		BL,
		RM,
		RMGPU,
		CDDT,
		PCDDT,
		OCDDT,
		CDDT2,
		PCDDT2,
		CDDTGPU,
		PCDDTGPU,
		GLT,
		UNKNOWN
	};

	Method which_method(std::string method) {
		if (method == "BresenhamsLine" || method == "bl") return BL;
		if (method == "RayMarching"    || method == "rm") return RM;
		if (method == "RayMarchingGPU" || method == "rmgpu") return RMGPU;
		if (method == "CDDTCast"       || method == "cddt") return CDDT;
		if (method == "PrunedCDDTCast" || method == "pcddt") return PCDDT;
		if (method == "CDDTCast2"      || method == "cddt2") return CDDT2;
		if (method == "PrunedCDDTCast2"|| method == "pcddt2") return PCDDT2;
		if (method == "CDDTCastGPU"    || method == "cddtgpu") return CDDTGPU;
		if (method == "PCDDTCastGPU"   || method == "pcddtgpu") return PCDDTGPU;
		if (method == "GiantLUTCast"   || method == "glt") return GLT;
		if (method == "OnlineCDDTCast" || method == "ocddt") return OCDDT;
		return UNKNOWN;
	}

	std::string abbrev(Method meth) {
		switch(meth)
		{
			case BL : return "bl";
			case RM : return "rm";
			case RMGPU : return "rmgpu";
			case CDDT : return "cddt";
			case PCDDT : return "pcddt";
			case CDDT2 : return "cddt2";
			case PCDDT2 : return "pcddt2";
			case CDDTGPU : return "cddtgpu";
			case PCDDTGPU : return "pcddtgpu";
			case OCDDT : return "ocddt";
			case GLT : return "glt";
			default : return "unknown";
		}
	}

	bool has(Method method, std::vector<Method> methods) {
		return std::find(methods.begin(), methods.end(),method)!=methods.end();
	}

	bool has(std::string substring, std::string str) {
		return str.find(substring) != std::string::npos;
	}

	bool has(std::string val, std::vector<std::string> vstr) {
		return std::find(vstr.begin(), vstr.end(),val)!=vstr.end();
	}

	std::vector<std::string> split(std::string in, char delim) {
		std::vector<std::string> result;
		std::stringstream ss(in);
		while( ss.good() )
		{
			std::string substr;
			std::getline( ss, substr, delim );
			result.push_back( substr );
		}
		return result;
	}

	double norminv(double q) {
		if(q == .5)
			return 0;

		q = 1.0 - q;

		double p = (q > 0.0 && q < 0.5) ? q : (1.0 - q);
		double t = sqrt(log(1.0 / pow(p, 2.0)));

		double c0 = 2.515517;
		double c1 = 0.802853;
		double c2 = 0.010328;

		double d1 = 1.432788;
		double d2 = 0.189269;
		double d3 = 0.001308;

		double x = t - (c0 + c1 * t + c2 * pow(t, 2.0)) /
					(1.0 + d1 * t + d2 * pow(t, 2.0) + d3 * pow(t, 3.0));

		if(q > .5)
		  x *= -1.0;

		return x;
	}

	template<typename T, typename U>
	struct is_same
	{
		static const bool value = false;
	};

	template<typename T>
	struct is_same<T, T>
	{
		static const bool value = true;
	};

	template<typename T, typename U>
	bool eqTypes() { return is_same<T, U>::value; }

	// http://stackoverflow.com/questions/311703/algorithm-for-sampling-without-replacement
	// Here's some code for sampling without replacement based on Algorithm 3.4.2S of Knuth's book Seminumeric Algorithms.
	class NonReplacementSampler
	{
	public:
		NonReplacementSampler() {
			rand = std::uniform_real_distribution<double>(0.0,1.0);
			generator.seed(clock());
		}
		~NonReplacementSampler() {}

		void sample(int populationSize, int sampleSize, std::vector<int> & samples) {
			int t = 0; // total input records dealt with
			int m = 0; // number of items selected so far
			double u;

			while (m < sampleSize) {
				// u = rand(generator);
				u = std::rand() / (float)RAND_MAX;
				if ((populationSize-t)*u >= sampleSize - m) t++;
				else {
					samples.push_back(t);
					t++; m++;
				}
			}
			// samples.push_back(1);
		}

		std::uniform_real_distribution<double> rand;
		std::default_random_engine generator;
	};

	class FastRand
	{
	public:
		FastRand() : FastRand(10000) {};
		FastRand(int n) : cache_size(n) {
			populate();
			repopulate_threshold = 1.0 / cache_size;
		}
		~FastRand(){};

		float rand() {
			// return std::rand() / (float)RAND_MAX;
			// float v = cache[i];
			if (i++>cache_size-1) i = 0;
			// if (v < repopulate_threshold) populate();
			return cache[i];
		}

		void populate() {
			// cache.empty();
			// for (int i = 0; i < cache_size; ++i) cache.push_back(std::rand() / (float)RAND_MAX);
			for (int i = 0; i < cache_size; ++i) cache[i] = std::rand() / (float)RAND_MAX;
		}
		int i = 0;
		int cache_size;
		float repopulate_threshold;
		// std::vector<float> cache;
		float cache[10000];
	};

	void serialize(std::vector<bool> &vals,std::stringstream* ss) {
		if (vals.size() == 0) {
			(*ss) << "[]";
			return; 
		}
		(*ss) << "[" << vals[0];
		for (int i = 1; i < vals.size(); ++i) {
			(*ss) << "," << vals[i];
		}
		(*ss) << "]";
	}

	void serialize(std::vector<float> &vals,std::stringstream* ss) {
		if (vals.size() == 0) {
			(*ss) << "[]";
			return; 
		}
		(*ss) << "[" << vals[0];
		for (int i = 1; i < vals.size(); ++i) {
			(*ss) << "," << vals[i];
		}
		(*ss) << "]";
	}

	std::string serialize(std::vector<float> &vals) {
		std::stringstream ss;
		serialize(vals,&ss);
		return ss.str();
	}

	void add_pixel(int x, int y, std::vector<int> &xs, std::vector<int> &ys) {
		xs.push_back(x); ys.push_back(y);
	}

	void draw_circle(int x0, int y0, int radius, std::vector<int> &xs, std::vector<int> &ys)
	{
	    int x = radius;
	    int y = 0;
	    int err = 0;
	 
	    while (x >= y)
	    {
	    add_pixel(x0 + x, y0 + y, xs, ys);
	    add_pixel(x0 + y, y0 + x, xs, ys);
	    add_pixel(x0 - y, y0 + x, xs, ys);
	    add_pixel(x0 - x, y0 + y, xs, ys);
	    add_pixel(x0 - x, y0 - y, xs, ys);
	    add_pixel(x0 - y, y0 - x, xs, ys);
	    add_pixel(x0 + y, y0 - x, xs, ys);
	    add_pixel(x0 + x, y0 - y, xs, ys);
	 
	    if (err <= 0)
	    {
	        y += 1;
	        err += 2*y + 1;
	    }
	 
	    if (err > 0)
	    {
	        x -= 1;
	        err -= 2*x + 1;
	    }
	    }
	}

	void radial_queries(int x0, int y0, int num_queries, std::vector<int> &xs, std::vector<int> &ys, std::vector<double> &ts) {
		double step = M_2PI / num_queries;
		double angle = 0;
		for (int i = 0; i < num_queries; ++i)
		{
			xs.push_back(x0);
			ys.push_back(y0);
			ts.push_back(angle);
			angle += step;
		}
	}

	template <int NodeSize = 128, class int_t = int, int precision = 16>
	class BTreeStructure
	{
		typedef btree::btree_set<int_t, std::less<int_t>, std::allocator<int_t>, NodeSize> tree_t;
	public:
		// float values are given, but internally these are converted to integers for
		// precision.
		BTreeStructure() {};
		~BTreeStructure() {};
		void insert(float key) {
			int value = key * precision;
			map.insert(value);
		};
		void remove(float key) {
			int value = key * precision;
			map.erase(value);
		};
		float next(float key) {
			int value = key * precision;
			typename tree_t::iterator it = map.lower_bound(value);
			if (it != map.end()) {
				return float(*it) / precision;
			} else {
				return -1.0;
			}
		};
		float prev(float key) {
			int value = key * precision;
			typename tree_t::iterator it = map.lower_bound(value);
			if (it != map.begin()) {
				it--;
				return float(*it) / precision;
			} else {
				return -1.0;
			}
		};

		size_t size() {
			return map.size();
		}
	private:
	    tree_t map;
	};


	template <int NodeSize = 128>
	class FloatBTreeStructure
	{
		typedef btree::btree_set<float, std::less<float>, std::allocator<float>, NodeSize> tree_t;
	public:
		// float values are given, but internally these are converted to integers for
		// precision.
		FloatBTreeStructure() {};
		~FloatBTreeStructure() {};
		void insert(float key) {
			map.insert(key);
		};
		void remove(float key) {
			map.erase(key);
		};
		float next(float key) {
			typename tree_t::iterator it = map.lower_bound(key);
			if (it != map.end()) {
				return *it;
			} else {
				return -1.0;
			}
		};
		float prev(float key) {
			typename tree_t::iterator it = map.lower_bound(key);
			if (it != map.begin()) {
				it--;
				return *it;
			} else {
				return -1.0;
			}
		};
	private:
	    tree_t map;
	};

	template <class int_t = int, int precision = 16>
	class SortedVector
	{
	public:
		SortedVector() {};
		~SortedVector() {};

		void insert(float key) {
			int value = key * precision;
			int index = std::upper_bound(vec.begin(), vec.end(), value) - vec.begin();

			vec.insert(vec.begin() + index, value);

			// if (index == 0) {
			// 	vec.insert(vec.begin(), value);
			// } else if (index == vec.size()) {
			// 	vec.push_back(value);
			// } else {
			// 	vec.insert(vec.begin() + index, value);
			// }
		};
		void remove(float key) {
			int value = key * precision;
			auto pr = std::equal_range(std::begin(vec), std::end(vec), value);
	    	vec.erase(pr.first, pr.second);
		};
		float next(float key) {
			int value = key * precision;
			int index = std::upper_bound(vec.begin(), vec.end(), value) - vec.begin();
			if (index < vec.size()) {
				return float(vec[index]) / precision;
			} else {
				return -1.0;
			}
		};
		float prev(float key) {
			int value = key * precision;
			int index = std::upper_bound(vec.begin(), vec.end(), value) - vec.begin();
			if (index > 0) {
				return float(vec[index-1]) / precision;
			} else {
				return -1.0;
			}
		};
	// private:
		std::vector<int_t> vec;
	};

	class SetStructure
	{
	public:
		// float values are given, but internally these are converted to integers for
		// precision.
		SetStructure() {

		};
		~SetStructure() {};
		void insert(float key) {
			int value = key * precision;
			map.insert(value);
		};
		void remove(float key) {
			int value = key * precision;
			map.erase(value);
		};
		float next(float key) {
			int value = key * precision;
			std::set<int>::iterator it = map.lower_bound(value);
			if (it != map.end()) {
				return float(*it) / precision;
			} else {
				// std::cout << "No larger element found!" << std::endl;
				return -1.0;
			}
		};
		float prev(float key) {
			int value = key * precision;
			std::set<int>::iterator it = map.lower_bound(value);
			if (it != map.begin()) {
				it--;
				return float(*it) / precision;
			} else {
				// std::cout << "No smaller element found!" << std::endl;
				return -1.0;
			}
		};
	private:
	    std::set<int> map;
	    int size = 0;
	    int precision = 16;
	};

	double ts(std::chrono::duration<double> diff) {
		return std::chrono::duration_cast<std::chrono::duration<double>>(diff).count();
	}

	template <class el_t>
	el_t median(std::vector<el_t> v)
	{
	    size_t n = v.size() / 2;
	    nth_element(v.begin(), v.begin()+n, v.end());
	    return v[n];
	}

} // namespace utils

#endif	/* _RANGE_UTILS_H_INCLUDED_ */
