#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class AddNFPGA;

class AddNFPGAInterface{
public:
	AddNFPGAInterface();
	void Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols);
	~AddNFPGAInterface();
private:
	AddNFPGA *_AddNFPGA;
}; 
#endif