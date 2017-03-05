#include <vector>


class RayMarchingCUDA
{
public:
	RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr);
	~RayMarchingCUDA();
	void calc_range_many(float *ins, float *outs, int num_casts);
private:
	float *d_ins;
	float *d_outs;
	float *d_distMap;
	int width;
	int height;
	float max_range;
};