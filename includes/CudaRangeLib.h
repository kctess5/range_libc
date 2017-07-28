#include <vector>

#ifndef ROS_WORLD_TO_GRID_CONVERSION
#define ROS_WORLD_TO_GRID_CONVERSION 1
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502
#endif

// class CUDARayCastMethod
// {
// public:
// 	CUDARayCastMethod(arguments);
// 	~CUDARayCastMethod();
// };

class RayMarchingCUDA
{
public:
	RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr);
	~RayMarchingCUDA();
	void calc_range_many(float *ins, float *outs, int num_casts);
	void numpy_calc_range(float *ins, float *outs, int num_casts);
	void numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles);

	void set_sensor_table(double *sensor_table, int table_width);

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	void set_conversion_params(float w_scale, float w_angle, float w_origin_x, 
		float w_origin_y, float w_sin_angle, float w_cos_angle) {
			inv_world_scale = 1.0 / w_scale; 
			world_scale = w_scale; 
			world_angle = w_angle;
			world_origin_x = w_origin_x;
			world_origin_y = w_origin_y;
			world_sin_angle = w_sin_angle;
			world_cos_angle = w_cos_angle;
			rotation_const = -1.0 * w_angle - 3.0*M_PI / 2.0;
			constants_set = true;
	}
	#endif
	
private:
	float *d_ins;
	float *d_outs;
	float *d_distMap;
	double *d_sensorTable;
	double *d_weights;
	int width;
	int height;
	int table_width;
	float max_range;

	bool allocated_weights = false;

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	float inv_world_scale;
	float world_scale;
	float world_angle;
	float world_origin_x;
	float world_origin_y;
	float world_sin_angle;
	float world_cos_angle;
	float rotation_const;
	bool constants_set = false;
	#endif
};

struct cddt_cast_data
{
	float *lut_bin;
	unsigned short lut_bin_width;
};

class CDDTCUDA
{
public:
	CDDTCUDA(std::vector<std::vector<bool> > grid, int w, int h, float mr, int theta_disc);
	~CDDTCUDA();
	int64_t init_buffers(float *compressed_lut_ptr, unsigned int *compressed_lut_index, unsigned short *lut_slice_widths, unsigned short *lut_bin_widths, int num_lut_els, int max_lut_w, float *lut_translations);
	void calc_range_many(float *ins, float *outs, int num_casts);

	void numpy_calc_range(float *ins, float *outs, int num_casts);
	void numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles);

	void set_sensor_table(double *sensor_table, int table_width);

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	void set_conversion_params(float w_scale, float w_angle, float w_origin_x, 
		float w_origin_y, float w_sin_angle, float w_cos_angle) {
			inv_world_scale = 1.0 / w_scale; 
			world_scale = w_scale; 
			world_angle = w_angle;
			world_origin_x = w_origin_x;
			world_origin_y = w_origin_y;
			world_sin_angle = w_sin_angle;
			world_cos_angle = w_cos_angle;
			rotation_const = -1.0 * w_angle - 3.0*M_PI / 2.0;
			constants_set = true;
	}
	#endif
protected:
	bool is_initialized;
	int theta_discretization;
	int max_lut_width;

	int width;
	int height;
	int table_width;
	float max_range;

	float *d_compressed_lut_ptr;
	float **d_compressed_lut_index;
	unsigned short *d_lut_slice_widths;
	unsigned short *d_lut_bin_widths;

	cddt_cast_data *d_cast_data;

	float *d_ins;
	float *d_outs;
	bool *d_map;

	double *d_sensorTable;

	#if ROS_WORLD_TO_GRID_CONVERSION == 1
	float inv_world_scale;
	float world_scale;
	float world_angle;
	float world_origin_x;
	float world_origin_y;
	float world_sin_angle;
	float world_cos_angle;
	float rotation_const;
	bool constants_set = false;
	#endif
};