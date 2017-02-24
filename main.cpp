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

#include "includes/RangeLib.h"
#include <gflags/gflags.h>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// DEFINE_bool(big_menu, true, "Include 'advanced' options in the menu listing");
DEFINE_string(method, "RayMarching",
	"Which range method to use, one of:\n  BresenhamsLine (or bl)\n  RayMarching (or rm)\n  CDDTCast (or cddt)\n  PrunedCDDTCast (or pcddt)\n  GiantLUTCast (or glt)\n");

DEFINE_string(map_path, "BASEMENT_MAP",
	"Path to map image, relative to current directory");

DEFINE_string(cddt_save_path, "",
	"Path to serialize CDDT data structure to.");

#define MAX_DISTANCE 500
#define NUM_RAYS 500
#define THETA_DISC 108
#define MB (1024.0*1024.0)
#ifdef BASEPATH
#define VERBOSE_LOG_PATH BASEPATH "/tmp"
#endif

// grid sample settings
#define GRID_STEP 10
#define GRID_RAYS 40
#define GRID_SAMPLES 1


using namespace ranges;
using namespace benchmark;

#ifdef VERBOSE_LOG_PATH
void save_log(std::stringstream &log, const char *fn) {
	std::cout << "...saving log to: " << fn << std::endl;
	std::ofstream file;  
	file.open(fn);
	file << log.str();
	file.close();
}
#endif

int main(int argc, char *argv[])
{
	// set usage message
	std::string usage("This library provides fast 2D ray casting on occupancy grid maps.  Sample usage:\n\n");
	usage += "   ";
	usage += argv[0];
	google::SetUsageMessage(usage);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::cout << "Running RangeLib benchmarks" << std::endl;
	OMap map = OMap(1,1);
	if (FLAGS_map_path == "BASEMENT_MAP") {
		#ifdef BASEPATH
		std::cout << "...Loading map" << std::endl;
		map = OMap(BASEPATH "/maps/basement_hallways_5cm.png");
		// #pragma GCC diagnostic push
		// #pragma GCC diagnostic ignored "-Wunused-result"
		// (volatile void) chdir(BASEPATH);
		// #pragma GCC diagnostic pop
		#else
		std::cout << "BASEPATH not defined, map paths may not resolve correctly." << std::endl;
		#endif
		
	} else {
		map = OMap(FLAGS_map_path);
	}

	#ifdef VERBOSE_LOG_PATH
	std::cout << "Saving logs to: " << VERBOSE_LOG_PATH "/" << std::endl;
	std::stringstream log; 
	#endif
	
	if (map.error()) return 1;

	std::vector<std::string> methods = utils::split(FLAGS_method, ',');

	if (utils::has("BresenhamsLine", methods) || utils::has("bl", methods)) {
		std::cout << "\n...Loading range method: BresenhamsLine" << std::endl;
		Benchmark<BresenhamsLine> mark = Benchmark<BresenhamsLine>(BresenhamsLine(map, MAX_DISTANCE));
		std::cout << "...Running grid benchmark" << std::endl;
		
		#ifdef VERBOSE_LOG_PATH
		log.str("");
		mark.set_log(&log);
		#endif
		// run the benchmark
		mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		#ifdef VERBOSE_LOG_PATH
		save_log(log, VERBOSE_LOG_PATH "/bl.csv");
		#endif
	}

	if (utils::has("RayMarching", methods) || utils::has("rm", methods)) {
		std::cout << "\n...Loading range method: RayMarching" << std::endl;
		Benchmark<RayMarching> mark = Benchmark<RayMarching>(RayMarching(map, MAX_DISTANCE));
		std::cout << "...Running grid benchmark" << std::endl;
		#ifdef VERBOSE_LOG_PATH
		log.str("");
		mark.set_log(&log);
		#endif
		// run the benchmark
		mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		#ifdef VERBOSE_LOG_PATH
		save_log(log, VERBOSE_LOG_PATH "/rm.csv");
		#endif
	}

	if (utils::has("CDDTCast", methods) || utils::has("cddt", methods)
		|| utils::has("PrunedCDDTCast", methods) || utils::has("pcddt", methods)) {
		CDDTCast rc = CDDTCast(map, MAX_DISTANCE, THETA_DISC);

		if (utils::has("CDDTCast", methods) || utils::has("cddt", methods)) {
			std::cout << "\n...Loading range method: CDDTCast" << std::endl;
			std::cout << "...lut size (MB): " << rc.lut_size() / MB << std::endl;
			Benchmark<CDDTCast> mark = Benchmark<CDDTCast>(rc);
			std::cout << "...Running grid benchmark" << std::endl;
			#ifdef VERBOSE_LOG_PATH
			log.str("");
			mark.set_log(&log);
			#endif
			// run the benchmark
			mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			#ifdef VERBOSE_LOG_PATH
			save_log(log, VERBOSE_LOG_PATH "/cddt.csv");
			#endif
		}

		if (utils::has("PrunedCDDTCast", methods) || utils::has("pcddt", methods)) {
			std::cout << "\n...Loading range method: PrunedCDDTCast" << std::endl;
			rc.prune(MAX_DISTANCE);
			std::cout << "...pruned lut size (MB): " << rc.lut_size() / MB << std::endl;
			Benchmark<CDDTCast> mark = Benchmark<CDDTCast>(rc);
			std::cout << "...Running grid benchmark" << std::endl;
			#ifdef VERBOSE_LOG_PATH
			log.str("");
			mark.set_log(&log);
			#endif
			// run the benchmark
			mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			#ifdef VERBOSE_LOG_PATH
			save_log(log, VERBOSE_LOG_PATH "/pcddt.csv");
			#endif
		}

		if (!FLAGS_cddt_save_path.empty()) {\
			std::cout << "...saving CDDT to:" << FLAGS_cddt_save_path<< std::endl;
			std::stringstream cddt_serialized;
			rc.serializeJson(&cddt_serialized);
			save_log(cddt_serialized, FLAGS_cddt_save_path.c_str());
		}
	}

	if (utils::has("GiantLUTCast", methods) || utils::has("glt", methods)) {
		std::cout << "\n...Loading range method: GiantLUTCast" << std::endl;
		GiantLUTCast glt = GiantLUTCast(map, MAX_DISTANCE, THETA_DISC);
		Benchmark<GiantLUTCast> mark = Benchmark<GiantLUTCast>(glt);
		std::cout << "...lut size (MB): " << glt.lut_size() / MB << std::endl;
		std::cout << "...Running grid benchmark" << std::endl;
		#ifdef VERBOSE_LOG_PATH
		log.str("");
		mark.set_log(&log);
		#endif
		// run the benchmark
		mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		#ifdef VERBOSE_LOG_PATH
		save_log(log, VERBOSE_LOG_PATH "/glt.csv");
		#endif
	}

	std::cout << "done" << std::endl;
	return 0;
}