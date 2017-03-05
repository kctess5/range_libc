#include "includes/CudaRangeLib.h"
// #include "includes/RangeUtils.h"


#include <cuda.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#define DIST_THRESHOLD 0.0
#define STEP_COEFF 0.999

__device__ float distance(int x, int y, float *distMap, int width, int height) {
	return distMap[x * height + y];
}

__global__ void cuda_ray_marching(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_casts) {
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

RayMarchingCUDA::RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr) 
	: width(w), height(h), max_range(mr) {
	cudaMalloc((void **)&d_ins, sizeof(float) * CHUNK_SIZE * 3);
	cudaMalloc((void **)&d_outs, sizeof(float) * CHUNK_SIZE);
	cudaMalloc((void **)&d_distMap, sizeof(float) * width * height);

	// convert vector format to raw float array, y axis is quickly changing dimension
	float raw_grid[width*height];
	for (int i = 0; i < width; ++i) std::copy(grid[i].begin(), grid[i].end(), &raw_grid[i*height]);

	cudaMemcpy(d_distMap, raw_grid, width*height*sizeof(float), cudaMemcpyHostToDevice);
}

RayMarchingCUDA::~RayMarchingCUDA() {
	cudaFree(d_ins); cudaFree(d_outs); cudaFree(d_distMap);
}

// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::calc_range_many(float *ins, float *outs, int num_casts) {
	// copy queries to GPU buffer
	// std::cout << "Copying memory to device: " << num_casts << std::endl;
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU
	cuda_ray_marching<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range, num_casts);
	// copy results back to CPU
	// std::cout << "Copying results to cpu" << std::endl;
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}