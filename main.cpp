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

DEFINE_string(which_benchmark, "",
	"Which benchmark to run, one of:\n  random\n  grid\n");

DEFINE_string(query, "",
	"Query point x,y,theta to ray cast from. example: --query=0,0,3.14");

DEFINE_string(trace_path, "", "Path to output trace map of memory access pattern. Works for Bresenham's Line or Ray Marching.");
DEFINE_string(lut_slice_path, "", "Path to output a slice of the LUT.");
DEFINE_string(lut_slice_theta, "1.57", "Which LUT slice to output");
DEFINE_string(serialize_path, "", "Path to output a serialized data structure.");

#define MAX_DISTANCE 500
#define THETA_DISC 108
#define MB (1024.0*1024.0)

// this trick is to avoid compiler sadness about the quotations for the BASEPATH define
#define Q(x) #x
#define QUOTE(x) Q(x)


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

template <class CAST_T>
CAST_T construct_continuous_method(OMap map, float max_dist, std::string map_path, bool serialized_input) {
	if (serialized_input) {
		CAST_T march;
		march.deserialize(map_path);
		return march;
	} else {
		return CAST_T(map, max_dist);
	}
}

template <class CAST_T>
CAST_T construct_discrete_method(OMap map, float max_dist, int theta_disc, std::string map_path, bool serialized_input) {
	if (serialized_input) {
		CAST_T rc;
		rc.deserialize(map_path);
		return rc;
	} else {
		return CAST_T(map, max_dist, theta_disc);
	}
}

bool fleq(float a, float b) {
	return fabs(a - b) < 0.001;
}

bool sanity_check() {
	OMap map = OMap(QUOTE(BASEPATH) "/maps/basement_hallways_5cm.png");

	int width = map.width;
	int height = map.height;

	CDDTCast rc = CDDTCast(map, 500, 108);
	CDDTCastGPU rcgpu = CDDTCastGPU(map, 500, 108);
	RayMarching rm = RayMarching(map, 500);
	RayMarchingGPU rmgpu = RayMarchingGPU(map, 500);

	float *samples = new float[RANDOM_SAMPLES*3];
	float *outs = new float[RANDOM_SAMPLES];
	float *outs2 = new float[RANDOM_SAMPLES];
			
	Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);

	rcgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);


	rmgpu.calc_range_many(samples, outs2, RANDOM_SAMPLES);

	int num_bad = 0;
	bool sane = true;
	for (int i = 0; i < RANDOM_SAMPLES; ++i)
	{
		float rng = rc.calc_range(samples[i*3+0], samples[i*3+1], samples[i*3+2]);
		bool sall_good = fleq(outs[i], rng);


		float gnd = rm.calc_range(samples[i*3+0], samples[i*3+1], samples[i*3+2]);
		// bool sall_good = fleq(outs2[i], gnd);
		sane = sane && sall_good;

		if (!sall_good && num_bad < 100) {
			
			std::cout << "Expected: " << rng << "   got: " << outs[i] << "  gnd: " << gnd << ", " << outs2[i] << std::endl;
		}

		if (!sall_good)
		{
			num_bad += 1;
		}
	}
	std::cout << "num_bad: " << num_bad << std::endl;
	return sane;
	// create CDDT method
	// create CDDTGPU method
	// get a shit ton of samples
	// compare outputs, they should be equivalent
}

int main(int argc, char *argv[]) {
	bool is_sane = sanity_check();

	std::cout << "is_sane: " << is_sane << std::endl;

	// exit(1);


	// set usage message
	std::string usage("This library provides fast 2D ray casting on occupancy grid maps.  Sample usage:\n\n");
	usage += "   ";
	usage += argv[0];
	google::SetUsageMessage(usage);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::vector<std::string> map_path_parts = utils::split(FLAGS_map_path, '/');
	std::string map_name = map_path_parts[map_path_parts.size()-1];
	std::cout << "MAP NAME: " << map_name << std::endl;
	std::vector<std::string> map_name_parts = utils::split(map_name, '.');
	bool SERIALIZED_INPUT = (map_name_parts[map_name_parts.size()-1] == "serialized");

	std::cout << "Running RangeLib benchmarks" << std::endl;
	OMap map = OMap(1,1);

	if (SERIALIZED_INPUT) {
		// std::cout << "SERIALIZED" << std::endl;
	} else if (FLAGS_map_path == "BASEMENT_MAP") {
		#ifdef BASEPATH
		std::cout << "...Loading map" << std::endl;
		map = OMap(QUOTE(BASEPATH) "/maps/basement_hallways_5cm.png");
		#else
		std::cout << "BASEPATH not defined, map paths may not resolve correctly." << std::endl;
		#endif		
	} else {
		map = OMap(FLAGS_map_path);
	}

	bool DO_SERIALIZE = (FLAGS_serialize_path != "" && !SERIALIZED_INPUT);
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

	float query_y, query_x, query_t;
	if (!FLAGS_trace_path.empty()) {
		std::vector<std::string> query = utils::split(FLAGS_query, ',');
		query_x = std::stof(query[0]);
		query_y = std::stof(query[1]);
		query_t = std::stof(query[2]);
	}

	std::vector<std::string> method_strs = utils::split(FLAGS_method, ',');
	std::vector<utils::Method> methods;
	std::cout << "methods: ";
	for (int i = 0; i < method_strs.size(); ++i) {
		methods.push_back(utils::which_method(method_strs[i]));
		std::cout << utils::abbrev(methods[i]);
		if (i != method_strs.size()-1) std::cout <<  ", ";
	}
		
	std::cout << std::endl;

	if (utils::has(utils::BL, methods)) {
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

		if (!FLAGS_query.empty()) {
			std::cout << "...querying pose:" << FLAGS_query << std::endl;
			std::cout << "...   range: " << bl.calc_range(query_x,query_y,query_t) << std::endl;
		}

		if (!FLAGS_trace_path.empty()) {
			#if _MAKE_TRACE_MAP == 1
			std::cout << "...saving trace to:" << FLAGS_trace_path << std::endl;
			bl.getMap()->saveTrace(FLAGS_trace_path);
			// mark.getMap()->saveTrace(FLAGS_trace_path);
			#else
			std::cout << "...CANNOT SAVE TRACE, first #define _MAKE_TRACE_MAP 1 to generate trace." << std::endl;
			#endif
		}
		
		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/bl.csv");
		}
	}

	if (utils::has(utils::RM, methods)) {
		std::cout << "\n...Loading range method: RayMarching" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();
		
		RayMarching rm = construct_continuous_method<RayMarching>(map, MAX_DISTANCE, FLAGS_map_path, SERIALIZED_INPUT);
		
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);
		Benchmark<RayMarching> mark = Benchmark<RayMarching>(rm);
		std::cout << "...construction time: " << construction_dur.count() << std::endl;
		std::cout << "...memory usage (MB): " << rm.memory() / MB << std::endl;
		std::cout << "...Running grid benchmark" << std::endl;


		if (DO_SERIALIZE) {
			rm.serialize(FLAGS_serialize_path + map_name + ".rm.serialized");
		}

		if (DO_LOG) {
			tlog.str("");
			mark.set_log(&tlog);
			summary << "rm," << construction_dur.count() << "," << rm.memory() << std::endl;
		}
		if (FLAGS_which_benchmark == "grid")
			mark.grid_sample2(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
		else if (FLAGS_which_benchmark == "random")
			mark.random_sample(RANDOM_SAMPLES);

		if (!FLAGS_query.empty()) {
			std::cout << "...querying pose:" << FLAGS_query << std::endl;
			std::cout << "...   range: " << rm.calc_range(query_x,query_y,query_t) << std::endl;
		}

		if (!FLAGS_trace_path.empty()) {
			#if _MAKE_TRACE_MAP == 1
			std::cout << "...saving trace to:" << FLAGS_trace_path << std::endl;
			rm.getMap()->saveTrace(FLAGS_trace_path);
			// mark.getMap()->saveTrace(FLAGS_trace_path);
			#else
			std::cout << "...CANNOT SAVE TRACE, first #define _MAKE_TRACE_MAP 1 to generate trace." << std::endl;
			#endif
		}


		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/rm.csv");
		}
	}

	if (utils::has(utils::CDDT, methods) || utils::has(utils::PCDDT, methods)) {
		auto construction_start = std::chrono::high_resolution_clock::now();

		CDDTCast rc = construct_discrete_method<CDDTCast>(map, MAX_DISTANCE, THETA_DISC, FLAGS_map_path, SERIALIZED_INPUT);

		// CDDTCast rc = CDDTCast(map, MAX_DISTANCE, THETA_DISC);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);

		double const_dur = 0;
		if (utils::has(utils::CDDT, methods)) {
			std::cout << "\n...Loading range method: CDDTCast" << std::endl;
			std::cout << "...construction time: " << construction_dur.count() << std::endl;
			std::cout << "...lut size (MB): " << rc.memory() / MB << std::endl;
			Benchmark<CDDTCast> mark = Benchmark<CDDTCast>(rc);
			std::cout << "...Running grid benchmark" << std::endl;
			if (DO_LOG) {
				tlog.str("");
				mark.set_log(&tlog);
				summary << "cddt," << construction_dur.count() << "," << rc.memory() << std::endl;
			}
			if (FLAGS_which_benchmark == "grid") {
				mark.grid_sample2(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
				mark.grid_sample2(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			}
				
			else if (FLAGS_which_benchmark == "random")
				mark.random_sample(RANDOM_SAMPLES);
			if (DO_LOG) {
				save_log(tlog, FLAGS_log_path+"/cddt.csv");
			}
		}

		if (utils::has(utils::PCDDT, methods)) {
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
				mark.grid_sample2(GRID_STEP, GRID_RAYS, GRID_SAMPLES);
			else if (FLAGS_which_benchmark == "random")
				mark.random_sample(RANDOM_SAMPLES);
			if (DO_LOG) {
				save_log(tlog, FLAGS_log_path+"/pcddt.csv");
			}
		}

		if (DO_SERIALIZE) {
			rc.serialize(FLAGS_serialize_path + map_name + ".cddt.serialized");
		}

		if (!FLAGS_cddt_save_path.empty()) {\
			std::cout << "...saving CDDT to:" << FLAGS_cddt_save_path<< std::endl;
			std::stringstream cddt_serialized;
			rc.serializeJson(&cddt_serialized);
			save_log(cddt_serialized, FLAGS_cddt_save_path.c_str());
		}
	}

	if (utils::has(utils::GLT, methods)) {
		std::cout << "\n...Loading range method: GiantLUTCast" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();

		GiantLUTCast glt = construct_discrete_method<GiantLUTCast>(map, MAX_DISTANCE, THETA_DISC, FLAGS_map_path, SERIALIZED_INPUT);

		// GiantLUTCast glt = GiantLUTCast(map, MAX_DISTANCE, THETA_DISC);
		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);

		if (DO_SERIALIZE) {
			glt.serialize(FLAGS_serialize_path + map_name + ".glt.serialized");
		}

		Benchmark<GiantLUTCast> mark = Benchmark<GiantLUTCast>(glt);
		std::cout << "...lut size (MB): " << glt.memory() / MB << std::endl;
		std::cout << "...construction time: " << construction_dur.count() << std::endl;
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

		if (!FLAGS_lut_slice_path.empty()) {
			std::cout << "...saving LUT slice theta="  << FLAGS_lut_slice_theta << " to: " << FLAGS_lut_slice_path << std::endl;
			float theta = std::stof(FLAGS_lut_slice_theta);
			glt.get_slice(theta)->save(FLAGS_lut_slice_path);
		}
		if (DO_LOG) {
			save_log(tlog, FLAGS_log_path+"/glt.csv");
		}
	}

	if (utils::has(utils::RMGPU, methods)) {
		#if USE_CUDA == 1
		std::cout << "\n...Loading range method: RayMarchingGPU" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();

		RayMarchingGPU rmgpu = construct_continuous_method<RayMarchingGPU>(map, MAX_DISTANCE, FLAGS_map_path, SERIALIZED_INPUT);

		int width = rmgpu.getMap()->width;
		int height = rmgpu.getMap()->height;

		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);
		std::cout << "...construction time: " << construction_dur.count() << std::endl;

		if (FLAGS_which_benchmark == "grid") {
			int num_samples = Benchmark<RayMarching>::num_grid_samples(GRID_STEP, GRID_RAYS, GRID_SAMPLES, width, height);
			float *samples = new float[num_samples*3];
			float *outs = new float[num_samples];
			

			Benchmark<RayMarching>::get_grid_samples(samples, GRID_STEP, GRID_RAYS, GRID_SAMPLES, width, height);
			// warmup
			rmgpu.calc_range_many(samples, outs, num_samples);

			auto mark_start = std::chrono::high_resolution_clock::now();
			rmgpu.calc_range_many(samples, outs, num_samples);
			auto mark_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> mark_dur = 
				std::chrono::duration_cast<std::chrono::duration<double>>(mark_end - mark_start);
			std::cout << "...benchmark time: " << mark_dur.count() << std::endl;
			std::cout << ".....avg time per ray cast: " << mark_dur.count() / num_samples << std::endl;
			std::cout << ".....rays cast: " << num_samples << std::endl;

			// print first few outputs for sanity checking
			for (int i = 0; i < 10; ++i)
				std::cout << outs[i] << std::endl;
		}
		
		if (FLAGS_which_benchmark == "random") {
			float *samples = new float[RANDOM_SAMPLES*3];
			float *outs = new float[RANDOM_SAMPLES];
			
			// warm up
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			rmgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			rmgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			auto mark_start = std::chrono::high_resolution_clock::now();
			rmgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			auto mark_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> mark_dur = 
				std::chrono::duration_cast<std::chrono::duration<double>>(mark_end - mark_start);
			std::cout << "...benchmark time: " << mark_dur.count() << std::endl;
			std::cout << ".....avg time per ray cast: " << mark_dur.count() / RANDOM_SAMPLES << std::endl;
			std::cout << ".....rays cast: " << RANDOM_SAMPLES << std::endl;

			// RayMarching rm = RayMarching(map, MAX_DISTANCE);
			// for (int i = 0; i < RANDOM_SAMPLES; ++i) {
			// 	if (std::abs(outs[i] - rm.calc_range(samples[3*i], samples[3*i+1], samples[3*i+2])) > 2) {
			// 		std::cout << "x: " << samples[3*i] << " y: " << samples[3*i+1] << " t: " << samples[3*i+2] << std::endl;
			// 		std::cout << "   expected: " << rm.calc_range(samples[3*i], samples[3*i+1], samples[3*i+2]) << std::endl;
			// 		std::cout << "   got: " << outs[i] << std::endl;
			// 	}
			// }
		}

		#else
		std::cout << "\nNot compiled with CUDA enabled, please enable flag -DWITH_CUDA=ON to run RayMarchingGPU benchmarks." << std::endl;
		#endif
	}

	if (utils::has(utils::CDDTGPU, methods) || utils::has(utils::PCDDTGPU, methods)) {
		#if USE_CUDA == 1
		std::cout << "\n...Loading range method: CDDTCastGPU" << std::endl;
		auto construction_start = std::chrono::high_resolution_clock::now();

		CDDTCastGPU cddtgpu = construct_discrete_method<CDDTCastGPU>(map, MAX_DISTANCE, THETA_DISC, FLAGS_map_path, SERIALIZED_INPUT);

		int width = cddtgpu.getMap()->width;
		int height = cddtgpu.getMap()->height;

		auto construction_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> construction_dur = 
			std::chrono::duration_cast<std::chrono::duration<double>>(construction_end - construction_start);
		std::cout << "...construction time: " << construction_dur.count() << std::endl;

		if (utils::has(utils::PCDDTGPU, methods)) {
			std::cout << "pruning" << std::endl;
			cddtgpu.prune(MAX_DISTANCE);
		}

		if (FLAGS_which_benchmark == "grid") {
			int num_samples = Benchmark<RayMarching>::num_grid_samples(GRID_STEP, GRID_RAYS, GRID_SAMPLES, width, height);
			float *samples = new float[num_samples*3];
			float *outs = new float[num_samples];
			
			Benchmark<RayMarching>::get_grid_samples(samples, GRID_STEP, GRID_RAYS, GRID_SAMPLES, width, height);
			// warmup
			cddtgpu.calc_range_many(samples, outs, num_samples);

			auto mark_start = std::chrono::high_resolution_clock::now();
			cddtgpu.calc_range_many(samples, outs, num_samples);
			auto mark_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> mark_dur = 
				std::chrono::duration_cast<std::chrono::duration<double>>(mark_end - mark_start);
			std::cout << "...benchmark time: " << mark_dur.count() << std::endl;
			std::cout << ".....avg time per ray cast: " << mark_dur.count() / num_samples << std::endl;
			std::cout << ".....rays cast: " << num_samples << std::endl;

			// print first few outputs for sanity checking
			for (int i = 0; i < 10; ++i)
				std::cout << outs[i] << std::endl;
		}
		
		if (FLAGS_which_benchmark == "random") {
			float *samples = new float[RANDOM_SAMPLES*3];
			float *outs = new float[RANDOM_SAMPLES];
			
			// warm up
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			cddtgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			cddtgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			
			Benchmark<RayMarching>::get_random_samples(samples, RANDOM_SAMPLES, width, height);
			auto mark_start = std::chrono::high_resolution_clock::now();
			cddtgpu.calc_range_many(samples, outs, RANDOM_SAMPLES);
			auto mark_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> mark_dur = 
				std::chrono::duration_cast<std::chrono::duration<double>>(mark_end - mark_start);
			std::cout << "...benchmark time: " << mark_dur.count() << std::endl;
			std::cout << ".....avg time per ray cast: " << mark_dur.count() / RANDOM_SAMPLES << std::endl;
			std::cout << ".....rays cast: " << RANDOM_SAMPLES << std::endl;
		}

		#else
		std::cout << "\nNot compiled with CUDA enabled, please enable flag -DWITH_CUDA=ON to run CDDTCastGPU benchmarks." << std::endl;
		#endif
	}

	if (DO_LOG) {
		save_log(summary, FLAGS_log_path+"/summary.csv");
	}

	std::cout << "done" << std::endl;
	return 0;
}