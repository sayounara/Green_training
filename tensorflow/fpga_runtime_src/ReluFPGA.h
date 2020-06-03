#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class ReluFPGA;

class ReluFPGAInterface{
public:
	ReluFPGAInterface();
	void Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols);
	~ReluFPGAInterface();
private:
	ReluFPGA *_ReluFPGA;
}; 
#endif