#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class Conv2DFPGA;

class Conv2DFPGAInterface{
public:
	Conv2DFPGAInterface();
	void Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols);
	~Conv2DFPGAInterface();
private:
	Conv2DFPGA *_Conv2DFPGA;
}; 
#endif