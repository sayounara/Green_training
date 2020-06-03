#ifndef MATCONV_FPGAImp_H
#define MATCONV_FPGAImp_H 

#define BLOCK_SIZE 256

class MatConvFPGAImp;

class MatConvFPGAImpInterface{
public:
	MatConvFPGAImpInterface();
	void Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_);
	~MatConvFPGAImpInterface();
private:
	MatConvFPGAImp *_MatConvFPGAImp;
}; 
#endif
