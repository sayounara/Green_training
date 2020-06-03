#ifndef MATCONV_FPGAImp_H
#define MATCONV_FPGAImp_H 

#define BLOCK_SIZE 256

class Conv2DFPGAImp;

class Conv2DFPGAImpInterface{
public:
	Conv2DFPGAImpInterface();
	void Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols);
	~Conv2DFPGAImpInterface();
private:
	Conv2DFPGAImp *_Conv2DFPGAImp;
}; 
#endif