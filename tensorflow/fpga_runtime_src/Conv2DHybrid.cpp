#include <stdio.h>
//#include <iostream>
//#include <type_traits>
#include <stdlib.h>
//#include <math.h>
#include "Conv2DGPU.h"
#include "Conv2DFPGAImp.h"
#include "AOCLUtils/aocl_utils.h"
#include "Conv2DFPGA.h"

class Conv2DFPGA
{
private:
	Conv2DFPGAImpInterface *_Conv2DFPGAImpInterface;
	Conv2DGPUInterface *_Conv2DGPUInterface;

public:
	Conv2DFPGA() {
		_Conv2DFPGAImpInterface=new Conv2DFPGAImpInterface();
		_Conv2DGPUInterface=new Conv2DGPUInterface();
	}

    void Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols)  {
    	_Conv2DFPGAImpInterface->Compute(in, filt, out, in_rows, in_cols, filt_rows, filt_cols);
    	_Conv2DGPUInterface->Compute(in, filt, out, in_rows, in_cols, filt_rows, filt_cols);
     }      //Implement function

	void CleanUp() {         //Clean OpenCL object
		_Conv2DFPGAImpInterface->~Conv2DFPGAImpInterface();
		_Conv2DGPUInterface->~Conv2DGPUInterface();

	}
};

Conv2DFPGAInterface::Conv2DFPGAInterface() {
	_Conv2DFPGA = new Conv2DFPGA();
} 

Conv2DFPGAInterface::~Conv2DFPGAInterface() {
	_Conv2DFPGA->CleanUp();
} 

void Conv2DFPGAInterface::Compute(const void *in, const void *filt, void *out, unsigned in_rows, unsigned in_cols, unsigned filt_rows, unsigned filt_cols) {
	_Conv2DFPGA->Compute(in, filt, out, in_rows, in_cols, filt_rows, filt_cols);
} 
