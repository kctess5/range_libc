#include "includes/CudaRangeLib.h"

#include <cuda.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>

#define USE_CONST_ANNOTATIONS 1

#define DIST_THRESHOLD 0.0
#define STEP_COEFF 0.999

__device__ float distance(int x, int y, const float *distMap, int width, int height) {
	return distMap[x * height + y];
}

#if USE_CONST_ANNOTATIONS == 1
__global__ void cuda_ray_marching(const float * ins, float * outs, const float * distMap, int width, int height, float max_range, int num_casts) {
#else
__global__ void cuda_ray_marching(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_casts) {
#endif
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_casts) return; 
	float x0 = ins[ind*3];
	float y0 = ins[ind*3+1];
	float theta = ins[ind*3+2];

	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	// int iters = 0;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}

		t += fmaxf(d * STEP_COEFF, 1.0);
		// iters ++;
	}
	outs[ind] = out;
}

__global__ void cuda_ray_marching_world_to_grid(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_casts, float world_origin_x, float world_origin_y, float world_scale, float inv_world_scale, float world_sin_angle, float world_cos_angle, float rotation_const) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_casts) return; 

	float x_world = ins[ind*3];
	float y_world = ins[ind*3+1];
	float theta_world = ins[ind*3+2];
	
	// convert x0,y0,theta from world to grid space coordinates
	float x0 = (x_world - world_origin_x) * inv_world_scale;
	float y0 = (y_world - world_origin_y) * inv_world_scale;
	float temp = x0;
	x0 = world_cos_angle*x0 - world_sin_angle*y0;
	y0 = world_sin_angle*temp + world_cos_angle*y0;
	float theta = -theta_world + rotation_const;

	// swap components
	temp = x0;
	x0 = y0;
	y0 = temp;

	// do ray casting
	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}

		t += fmaxf(d * STEP_COEFF, 1.0);
	}
	outs[ind] = out * world_scale;
}

__global__ void cuda_ray_marching_angles_world_to_grid(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_particles, int num_angles, float world_origin_x, float world_origin_y, float world_scale, float inv_world_scale, float world_sin_angle, float world_cos_angle, float rotation_const) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_angles*num_particles) return; 

	int angle_ind = fmodf( ind, num_angles );
	int particle_ind = (float) ind / (float) num_angles;

	float x_world = ins[particle_ind*3];
	float y_world = ins[particle_ind*3+1];
	float theta_world = ins[particle_ind*3+2];
	
	// convert x0,y0,theta from world to grid space coordinates
	float x0 = (x_world - world_origin_x) * inv_world_scale;
	float y0 = (y_world - world_origin_y) * inv_world_scale;
	float temp = x0;
	x0 = world_cos_angle*x0 - world_sin_angle*y0;
	y0 = world_sin_angle*temp + world_cos_angle*y0;
	float theta = -theta_world + rotation_const - ins[num_particles * 3 + angle_ind];

	// swap components
	temp = x0;
	x0 = y0;
	y0 = temp;

	// do ray casting
	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}
		t += fmaxf(d * STEP_COEFF, 1.0);
	}
	outs[ind] = out * world_scale;
}

__device__ int clamp(float val, float min, float max) {
	val = val>max?max:val;
	val = val<min?min:val;
	return (int)val;
}

// this should be optimized to use shared memory, otherwise the random read performance is not great
__global__ void cuda_eval_sensor_table(float * obs, float * ranges, double * outs, double * sensorTable, int rays_per_particle, int particles, float inv_world_scale, int max_range) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= rays_per_particle) return;

	int r = clamp(obs[ind] * inv_world_scale,0,max_range-1.0);
	for (int i = 0; i < particles; ++i)
	{
		int d = clamp(ranges[ind + ind * i],0,max_range-1.0);
		outs[ind+i*rays_per_particle] = sensorTable[r*max_range+d];
	}
}


#ifndef M_2PI
#define M_2PI 6.28318530718
#endif

#ifndef _EPSILON
#define _EPSILON 0.00001
#endif

#define CONST_MEMORY_SIZE 2048
__constant__ unsigned short constData[CONST_MEMORY_SIZE];

__device__ bool is_occupied(int x, int y, const bool *d_map, int height) {
	return d_map[x * height + y];
}

#if USE_CONST_ANNOTATIONS == 1
__global__ void cuda_cddt(
	const float * __restrict__ ins, 
	float * outs, 
	const bool * __restrict__ d_map, 
	int width, int height, float max_range, int num_casts, int theta_discretization, int max_lut_width, 
	const float * __restrict__ d_compressed_lut_ptr, 
	const float * const * __restrict__ d_compressed_lut_index, 
	const unsigned short * __restrict__ d_lut_slice_widths, 
	const unsigned short * __restrict__ d_lut_bin_widths) {
#else
__global__ void cuda_cddt(
	float * ins, 
	float * outs, 
	bool * d_map,
	int width, int height, float max_range, int num_casts, int theta_discretization, int max_lut_width, 
	float * d_compressed_lut_ptr, 
	float * * d_compressed_lut_index, 
	unsigned short * d_lut_slice_widths, 
	unsigned short * d_lut_bin_widths) {
#endif
	
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_casts) return; 
	float x = ins[ind*3];
	float y = ins[ind*3+1];
	float heading = -ins[ind*3+2];

	// discretize theta
	float theta = fmodf(heading, M_2PI);
	// fmod does not wrap the angle into the positive range, so this will fix that if necessary
	if (theta < 0.0) theta += M_2PI;

	bool is_flipped = false;
	if (theta >= M_PI) {
		is_flipped = true;
		theta -= M_PI;
	}

	int rounded = rintf(theta * theta_discretization / M_2PI);

	// this handles the special case where the theta rounds up and should wrap around
	if (rounded == theta_discretization >> 1) {
		rounded = 0;
		is_flipped = !is_flipped;
	}

	int angle_index = fmodf(rounded, theta_discretization);
	float discrete_angle = (angle_index * M_2PI) / ((float) theta_discretization);
	// project into lut space
	float cosangle;
	float sinangle;
	sincosf(discrete_angle, &sinangle, &cosangle);

	// compute LUT translation
	float left_top_corner_y     = height*cosangle;
	float right_bottom_corner_y = width*sinangle;
	float right_top_corner_y    = right_bottom_corner_y + left_top_corner_y;
	float min_corner_y = fminf(left_top_corner_y, fminf(right_top_corner_y, right_bottom_corner_y));
	float lut_translation = fminf(0.0, -1.0 * min_corner_y - _EPSILON);

	// float lut_translation = constData[angle_index];

	// do coordinate space projection
	float lut_space_x = x * cosangle - y * sinangle;
	float lut_space_y = (x * sinangle + y * cosangle) + lut_translation;

	// Convert a float to a signed integer in round-down mode.
	int lut_index = __float2int_rd(lut_space_y);

	// check d_lut_slice_widths if query is out of map
	// if (lut_index < 0 || lut_index >= d_lut_slice_widths[angle_index]) {
	if (lut_index < 0 || lut_index >= constData[angle_index]) {
		outs[ind] = max_range;
		return;
	}

	// get the lut bin using the lut index
	#if USE_CONST_ANNOTATIONS == 1
	const float *lut_bin = d_compressed_lut_index[angle_index*max_lut_width+lut_index];
	#else
	float *lut_bin = d_compressed_lut_index[angle_index*max_lut_width+lut_index];
	#endif

	// get the lut bin width using d_lut_bin_widths
	int lut_bin_width = d_lut_bin_widths[angle_index*max_lut_width+lut_index];

	if (lut_bin_width == 0) {
		outs[ind] = max_range;
		return;
	}

	int low = 0;
	int high = lut_bin_width - 1;

	if (is_flipped) {
		// the furthest entry is behind the query point
		if (lut_bin[low] > lut_space_x) {
			outs[ind] = max_range;
			return;
		}
		if (lut_bin[high]< lut_space_x) {
			outs[ind] = lut_space_x - lut_bin[high];
			return;
		}

		// TODO
		// if (map.grid[x][y]) { return 0.0; }
		// if (d_map[int(x) * height + int(y)]) {
		if (is_occupied(x, y, d_map, height)) {
			outs[ind] = 0.0;
			return;
		}

		for (int i = high; i >= 0; --i) {
			float obstacle_x = lut_bin[i];
			if (obstacle_x <= lut_space_x) {
				outs[ind] = lut_space_x - obstacle_x;
				return;
			}
		}
	} else {
		// the furthest entry is behind the query point
		if (lut_bin[high] < lut_space_x) {
			outs[ind] = max_range;
			return;
		}
		if (lut_bin[low] > lut_space_x) {
			outs[ind] = lut_bin[low] - lut_space_x;
			return;
		}

		// TODO
		// the query point is on top of a occupied pixel
		// this call is here rather than at the beginning, because it is apparently more efficient.
		// I presume that this has to do with the previous two return statements
		if (is_occupied(x, y, d_map, height)) {
			outs[ind] = 0.0;
			return;
		}

		// linear search for neighbor in lut bin
		for (int i = 0; i < lut_bin_width; ++i)
		{
			float obstacle_x = lut_bin[i];
			if (obstacle_x >= lut_space_x) {
				outs[ind] = obstacle_x - lut_space_x;
				return;
			}
		}
	}

	// check a few edge cases before search
	// 
	// 
	// 
	// 
	// make sure this pixel is not occupied in the source map
	// perform linear search through the lut bin to find neighbor in the lut
	// return the distance to the neighbor
	outs[ind] = -1.0;
}

void err_check() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
    	printf("Error: %s\n", cudaGetErrorString(err));
}

RayMarchingCUDA::RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr) 
	: width(w), height(h), max_range(mr) {
	cudaMalloc((void **)&d_ins, sizeof(float) * CHUNK_SIZE * 3);
	cudaMalloc((void **)&d_outs, sizeof(float) * CHUNK_SIZE);
	cudaMalloc((void **)&d_distMap, sizeof(float) * width * height);

	// convert vector format to raw float array, y axis is quickly changing dimension
	// float raw_grid[width*height];
	float *raw_grid = new float[width*height];
	for (int i = 0; i < width; ++i) std::copy(grid[i].begin(), grid[i].end(), &raw_grid[i*height]);

	cudaMemcpy(d_distMap, raw_grid, width*height*sizeof(float), cudaMemcpyHostToDevice);
	free(raw_grid);
}

RayMarchingCUDA::~RayMarchingCUDA() {
	cudaFree(d_ins); cudaFree(d_outs); cudaFree(d_distMap);
}

void RayMarchingCUDA::set_sensor_table(double *table, int t_w) {
	table_width = t_w;
	int table_size = sizeof(double) * table_width * table_width;
	cudaMalloc((void **)&d_sensorTable, table_size);
	cudaMemcpy(d_sensorTable, table, table_size, cudaMemcpyHostToDevice);
}

// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::calc_range_many(float *ins, float *outs, int num_casts) {
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU
	cuda_ray_marching<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range, num_casts);
	err_check();

	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::numpy_calc_range(float *ins, float *outs, int num_casts) {
	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU, have to pass coordinate space conversion constants
	cuda_ray_marching_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range,
		num_casts, world_origin_x, world_origin_y, world_scale, inv_world_scale, world_sin_angle, world_cos_angle, rotation_const);
	err_check();
	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#else
	std::cout << "GPU numpy_calc_range only works with ROS world to grid conversion enabled" << std::endl;
	#endif
}


// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles) {
	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_particles * 3,cudaMemcpyHostToDevice);
	// also copy angles to end of GPU buffer, this assumes there is enough space (which there should be)
	cudaMemcpy(&d_ins[num_particles * 3], angles, sizeof(float) * num_angles,cudaMemcpyHostToDevice);
	// execute queries on the GPU, have to pass coordinate space conversion constants
	cuda_ray_marching_angles_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range,
		num_particles, num_angles, world_origin_x, world_origin_y, world_scale, inv_world_scale, world_sin_angle, world_cos_angle, rotation_const);
	err_check();
	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_particles*num_angles,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#else
	std::cout << "GPU numpy_calc_range_angles only works with ROS world to grid conversion enabled" << std::endl;
	#endif
}

// void RayMarchingCUDA::calc_range_repeat_angles_eval_sensor_model(float * ins, float * angles, float * obs, double * weights, int num_particles, int num_angles) {
// 	#if ROS_WORLD_TO_GRID_CONVERSION == 1
// 	std::cout << "Do not use calc_range_repeat_angles_eval_sensor_model for GPU, unimplemented" << std::endl;
// 	std::cout << "Instead use numpy_calc_range_angles followed by a standard sensor evaluation method." << std::endl;
// 	// if (!allocated_weights) {
// 	// 	cudaMalloc((void **)&d_weights, num_particles*num_angles*sizeof(double));
// 	// 	allocated_weights = true;
// 	// }
// 	// // copy queries to GPU buffer
// 	// cudaMemcpy(d_ins, ins, sizeof(float) * num_particles * 3,cudaMemcpyHostToDevice);
// 	// // also copy angles to end of GPU buffer, this assumes there is enough space (which there should be)
// 	// cudaMemcpy(&d_ins[num_particles * 3], angles, sizeof(float) * num_angles,cudaMemcpyHostToDevice);

// 	// // execute queries on the GPU, have to pass coordinate space conversion constants
// 	// cuda_ray_marching_angles_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, 
// 	// 	width, height, max_range, num_particles, num_angles, world_origin_x, world_origin_y, 
// 	// 	world_scale, inv_world_scale, world_sin_angle, world_cos_angle, rotation_const);

// 	// cudaMemcpy(d_ins, obs, sizeof(float) * num_angles,cudaMemcpyHostToDevice);

// 	// // read from sensor table
// 	// cuda_eval_sensor_table<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins, d_outs, d_weights, d_sensorTable, num_angles, num_particles, inv_world_scale, table_width);

// 	// cuda_eval_sensor_table<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins, d_outs, d_weights, d_sensorTable, num_angles, num_particles, inv_world_scale, table_width);
// 	// // cuda_eval_sensor_table(angles, d_sensorTable)
// 	// // multiplicatively accumulate weights on the GPU
	

// 	// // copy weights back to CPU
// 	// cudaMemcpy(weights,d_weights,sizeof(double)*num_particles,cudaMemcpyDeviceToHost);
// 	// cudaDeviceSynchronize();
// 	#else
// 	std::cout << "GPU numpy_calc_range_angles only works with ROS world to grid conversion enabled" << std::endl;
// 	#endif
// }






///////////////////////////////////////////////////////////



CDDTCUDA::CDDTCUDA(std::vector<std::vector<bool> > grid, int w, int h, float mr, int theta_disc) 
	: is_initialized(false), theta_discretization(theta_disc), width(w), height(h), max_range(mr) {
	cudaMalloc((void **)&d_ins, sizeof(float) * CHUNK_SIZE * 3);
	cudaMalloc((void **)&d_outs, sizeof(float) * CHUNK_SIZE);
	cudaMalloc((void **)&d_map, sizeof(bool) * width * height);

	// convert vector format to raw bool array, y axis is quickly changing dimension
	bool *raw_grid = new bool[width*height];
	for (int i = 0; i < width; ++i) std::copy(grid[i].begin(), grid[i].end(), &raw_grid[i*height]);
	cudaMemcpy(d_map, raw_grid, width*height*sizeof(bool), cudaMemcpyHostToDevice);
	free(raw_grid);

}
CDDTCUDA::~CDDTCUDA() {
	if (is_initialized) {
		cudaFree(d_compressed_lut_ptr);
		cudaFree(d_compressed_lut_index);
		cudaFree(d_lut_slice_widths);
		cudaFree(d_lut_bin_widths);
	}

	cudaFree(d_ins);
	cudaFree(d_outs);
	cudaFree(d_map);
}

void CDDTCUDA::init_buffers(float *compressed_lut_ptr, unsigned int *compressed_lut_index, unsigned short *lut_slice_widths, unsigned short *lut_bin_widths, int num_lut_els, int max_lut_w, float *lut_translations) {
	std::cout << "Initializing buffers on device..." << std::endl;
	max_lut_width = max_lut_w;
	if (is_initialized) {
		std::cout << "...freeing old buffers" << std::endl;
		cudaFree(d_compressed_lut_ptr);
		cudaFree(d_compressed_lut_index);
		cudaFree(d_lut_slice_widths);
		cudaFree(d_lut_bin_widths);
	}

	// allocate space on the device for the CDDT structure
	cudaMalloc((void **)&d_compressed_lut_ptr, num_lut_els*sizeof(float));
	cudaMalloc((void **)&d_compressed_lut_index, max_lut_width*theta_discretization*sizeof(float**));
	cudaMalloc((void **)&d_lut_slice_widths, theta_discretization*sizeof(unsigned short));
	cudaMalloc((void **)&d_lut_bin_widths, max_lut_width*theta_discretization*sizeof(unsigned short));

	// copy LUT translations
	// cudaMemcpyToSymbol(constData, lut_translations, theta_discretization*sizeof(float));
	cudaMemcpyToSymbol(constData, lut_slice_widths, theta_discretization*sizeof(unsigned short));

	// copy cddt data structure
	cudaMemcpy(d_compressed_lut_ptr, compressed_lut_ptr, num_lut_els*sizeof(float), cudaMemcpyHostToDevice);

	// copy size metadata
	cudaMemcpy(d_lut_slice_widths, lut_slice_widths, 
		theta_discretization*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lut_bin_widths, lut_bin_widths, 
		max_lut_width*theta_discretization*sizeof(unsigned short), cudaMemcpyHostToDevice);
	
	// build device pointer index on the host
	float **device_pointer_index = (float **) malloc(max_lut_width*theta_discretization*sizeof(float**));
	for (int i = 0; i < theta_discretization*max_lut_width; ++i) {
		device_pointer_index[i] = &d_compressed_lut_ptr[compressed_lut_index[i]];
	}

	// move the pointer index to the device
	cudaMemcpy(d_compressed_lut_index, device_pointer_index, 
		max_lut_width*theta_discretization*sizeof(float**), cudaMemcpyHostToDevice);

	is_initialized = true;
	free(device_pointer_index);
}

void CDDTCUDA::calc_range_many(float *ins, float *outs, int num_casts) {
	if (!is_initialized) {
		std::cout << "Must initialize GPU buffers before using calc_range_many" << std::endl;
		return;
	}
	// std::cout << "cuda calc range" << std::endl;

	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU
	cuda_cddt<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_map, width, height, max_range, num_casts, theta_discretization, max_lut_width, d_compressed_lut_ptr, d_compressed_lut_index, d_lut_slice_widths, d_lut_bin_widths);
	err_check();

	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}






