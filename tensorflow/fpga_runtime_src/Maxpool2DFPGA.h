#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class Maxpool2DFPGA;

class Maxpool2DFPGAInterface{
public:
	Maxpool2DFPGAInterface();
	void Compute(const int input_size, const int output_size, void *input_im, void *output_im);
	~Maxpool2DFPGAInterface();
private:
	Maxpool2DFPGA *_Maxpool2DFPGA;
}; 
#endif
