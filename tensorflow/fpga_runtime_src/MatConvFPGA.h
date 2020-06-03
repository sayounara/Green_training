#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class MatConvFPGA;

class MatConvFPGAInterface{
public:
	MatConvFPGAInterface();
	void Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_);
	~MatConvFPGAInterface();
private:
	MatConvFPGA *_MatConvFPGA;
}; 
#endif
