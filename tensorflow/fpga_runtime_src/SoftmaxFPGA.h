#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class SoftmaxFPGA;

class SoftmaxFPGAInterface{
public:
	SoftmaxFPGAInterface();
	void Compute(const void *in);
	~SoftmaxFPGAInterface();
private:
	SoftmaxFPGA *_SoftmaxFPGA;
}; 
#endif