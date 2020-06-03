#include <stdio.h>
//#include <iostream>
//#include <type_traits>
#include <stdlib.h>
#include "MatConvFPGA.h"
#include "MatConvFPGAImp.h"
#include "MatConvGPU.h"
#include "model.h"

class MatConvFPGA
{
private:
	MatConvFPGAImpInterface *_MatConvFPGAImpInterface;
	MatConvGPUInterface *_MatConvGPUInterface;

public:
	MatConvFPGA() {
		_MatConvFPGAImpInterface = new MatConvFPGAImpInterface();
		_MatConvGPUInterface = new MatConvGPUInterface();	
	}

    void Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_)  {
    	const char **kernel_candidates={"matrixMult256", "matrixMult128", "matrixMult64"};//optimal FPGA kernel candidates
		float estimated_power[]={25, 29, 32};
		float power_reduction=50;//50W power reduction on GPU
		float performance_model;
		float extra_boost_ratio=1.15;
    	// int compute_cost[]={};
    	for (int i = 0; i < 3; ++i)
    	{
    		float p_gpu=gpu_performance_modeling("matmul", A_height_, A_width_, B_width_)
    		float p_fpga=fpga_performance_modeling("matmul", kernel_candidates[i], A_height_, A_width_, B_width_);
    		if (p_fpga>=p_gpu && estimated_power[i]<power_reduction)
    		{
    			_MatConvFPGAImpInterface->Compute(matrix_a, matrix_b, matrix_out, A_height_ , A_width_, B_width_);
    			break;
    		}
    		else if(p_fpga<p_gpu && p_gpu/p_fpga <=1.15)
    		{
    			_MatConvFPGAImpInterface->Compute(matrix_a, matrix_b, matrix_out, A_height_ , A_width_, B_width_);
    			break;
    		}
    		else if(!is_on_criticalpath("matmul", A_height_, A_width_, B_width_) && estimated_power[i]<power_reduction)
    		{
    			_MatConvFPGAImpInterface->Compute(matrix_a, matrix_b, matrix_out, A_height_ , A_width_, B_width_);
    			break;
    		}
    		else
    		{
    			_MatConvGPUInterface->Compute(matrix_a, matrix_b, matrix_out, A_height_ , A_width_, B_width_);
    			break;
    		}
    	}
     }      //Implement function

	void CleanUp() {         //Clean OpenCL object

		_MatConvFPGAImpInterface->~MatConvFPGAImpInterface();
		_MatConvGPUInterface->~MatConvGPUInterface();

	}
};

MatConvFPGAInterface::MatConvFPGAInterface() {
	_MatConvFPGA = new MatConvFPGA();
} 

MatConvFPGAInterface::~MatConvFPGAInterface() {
	_MatConvFPGA->CleanUp();
} 

void MatConvFPGAInterface::Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_) {
	_MatConvFPGA->Compute(matrix_a, matrix_b, matrix_out, A_height_, A_width_, B_width_);
} 
