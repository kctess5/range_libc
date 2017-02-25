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

DEFINE_string(log_path, "",
	"Where to store high fidelity logs, does not save if not specified.");

DEFINE_string(which_benchmark, "random",
	"Which benchmark to run, one of:\n  random\n  grid\n");

#define MAX_DISTANCE 500
#define NUM_RAYS 500
#define THETA_DISC 108
#define MB (1024.0*1024.0)
// #ifdef BASEPATH
// #define FLAGS_log_path BASEPATH "/tmp/basement/"
// #endif

// grid sample settings
#define GRID_STEP 10
#define GRID_RAYS 40
#define GRID_SAMPLES 1
#define RANDOM_SAMPLES 200000

using namespace ranges;
using namespace benchmark;

void save_log(std::stringstream &log, const char *fn) {
	std::cout << "...saving log to: " << fn << std::endl;
	std::ofstream file;  
	file.open(fn);
	file << log.str();
	file.close();
}

void save_log(std::stringstream &log, std::string path) {
	save_log(log,path.c_str());
}

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

	bool DO_LOG = (FLAGS_log_path != "");
	std::stringstream tlog;
	std::stringstream summary; 
	if (DO_LOG) {
		std::cout << "Saving logs to: " << FLAGS_log_path << std::endl;
		summary << std::fixed;
	    summary << std::setprecision(9);
		summary << "method,construction_time,memory_bytes" << std::endl;
	}

	if (map.error()) return 1;

	std::vector<std::string> methods = utils::split(FLAGS_method, ',');

	if (utils::has("BresenhamsLine", methods) || utils::has("bl", methods)) {
		std::cout << "\n...Loading range method: BresenhamsLine" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();
		BresenhamsLine bl = BresenhamsLine(map, MAX_DISTANCE);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);
		
		std::cout << "...Running grid benchmark" << std::endl;
		Benchmark<BresenhamsLine> mark = Benchmark<BresenhamsLine>(bl);
		
		if (DO_LOG) {
			tlog.str("");
			mark.set_log(&tlog);
			summary << "bl," << construction_dur.count() << "," << bl.memory() << std::endl;
		}
		// run the benchmark
		if (FLAGS_which_benchmark == "grid")
			mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		else if (FLAGS_which_benchmark == "random")
			mark.random_sample(RANDOM_SAMPLES);
		
		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/bl.csv");
		}
	}

	if (utils::has("RayMarching", methods) || utils::has("rm", methods)) {
		std::cout << "\n...Loading range method: RayMarching" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();
		RayMarching rm = RayMarching(map, MAX_DISTANCE);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);
		Benchmark<RayMarching> mark = Benchmark<RayMarching>(rm);
		std::cout << "...Running grid benchmark" << std::endl;
		if (DO_LOG) {
			tlog.str("");
			mark.set_log(&tlog);
			summary << "rm," << construction_dur.count() << "," << rm.memory() << std::endl;
		}
		if (FLAGS_which_benchmark == "grid")
			mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		else if (FLAGS_which_benchmark == "random")
			mark.random_sample(RANDOM_SAMPLES);
		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/rm.csv");
		}
	}

	if (utils::has("CDDTCast", methods) || utils::has("cddt", methods)
		|| utils::has("PrunedCDDTCast", methods) || utils::has("pcddt", methods)) {
		auto construction_start = std::chrono::high_resolution_clock::now();
		CDDTCast rc = CDDTCast(map, MAX_DISTANCE, THETA_DISC);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);

		double const_dur = 0;
		if (utils::has("CDDTCast", methods) || utils::has("cddt", methods)) {
			std::cout << "\n...Loading range method: CDDTCast" << std::endl;
			std::cout << "...lut size (MB): " << rc.memory() / MB << std::endl;
			Benchmark<CDDTCast> mark = Benchmark<CDDTCast>(rc);
			std::cout << "...Running grid benchmark" << std::endl;
			if (DO_LOG) {
				tlog.str("");
				mark.set_log(&tlog);
				summary << "cddt," << construction_dur.count() << "," << rc.memory() << std::endl;
			}
			if (FLAGS_which_benchmark == "grid")
				mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			else if (FLAGS_which_benchmark == "random")
				mark.random_sample(RANDOM_SAMPLES);
			if (DO_LOG) {
				save_log(tlog, FLAGS_log_path+"/cddt.csv");
			}
		}

		if (utils::has("PrunedCDDTCast", methods) || utils::has("pcddt", methods)) {
			std::cout << "\n...Loading range method: PrunedCDDTCast" << std::endl;
			

			auto prune_start = std::chrono::high_resolution_clock::now();
			rc.prune(MAX_DISTANCE);
			auto prune_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> prune_dur = 
				std::chrono::duration_cast<std::chrono::duration<double>>(prune_end - prune_start);


			std::cout << "...pruned lut size (MB): " << rc.memory() / MB << std::endl;
			Benchmark<CDDTCast> mark = Benchmark<CDDTCast>(rc);
			std::cout << "...Running grid benchmark" << std::endl;
			if (DO_LOG) {
				tlog.str("");
				mark.set_log(&tlog);
				summary << "pcddt," << construction_dur.count() + prune_dur.count() << "," << rc.memory() << std::endl;
			}
			if (FLAGS_which_benchmark == "grid")
				mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			else if (FLAGS_which_benchmark == "random")
				mark.random_sample(RANDOM_SAMPLES);
			if (DO_LOG) {
				save_log(tlog, FLAGS_log_path+"/pcddt.csv");
			}
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
		

		auto construction_start = std::chrono::high_resolution_clock::now();
		GiantLUTCast glt = GiantLUTCast(map, MAX_DISTANCE, THETA_DISC);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);


		Benchmark<GiantLUTCast> mark = Benchmark<GiantLUTCast>(glt);
		std::cout << "...lut size (MB): " << glt.memory() / MB << std::endl;
		std::cout << "...Running grid benchmark" << std::endl;
		if (DO_LOG) {
			tlog.str("");
			mark.set_log(&tlog);
			summary << "glt," << construction_dur.count() << "," << glt.memory() << std::endl;
		}
		if (FLAGS_which_benchmark == "grid")
			mark.grid_sample(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		else if (FLAGS_which_benchmark == "random")
			mark.random_sample(RANDOM_SAMPLES);
		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/glt.csv");
		}
	}

	if (DO_LOG) {
		save_log(summary, FLAGS_log_path+"/summary.csv");
	}

	std::cout << "done" << std::endl;
	return 0;
}