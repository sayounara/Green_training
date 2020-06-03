#ifndef MATCONV_GPU_H
#define MATCONV_GPU_H 

#define BLOCK_SIZE 256

class MatConvGPU;

class MatConvGPUInterface{
public:
	MatConvGPUInterface();
	void Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_);
	~MatConvGPUInterface();
private:
	MatConvGPU *_MatConvGPU;
}; 
#endif
