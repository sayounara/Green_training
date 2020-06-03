#ifndef MATCONV_FPGA_H
#define MATCONV_FPGA_H 

#define BLOCK_SIZE 256

class BatchnormFPGA;

class BatchnormFPGAInterface{
public:
	BatchnormFPGAInterface();
	void Compute(void *pMaps, void *pScale, void *pOffset, void *pOutput);
	~BatchnormFPGAInterface();
private:
	BatchnormFPGA *_BatchnormFPGA;
}; 
#endif
