#include <stdio.h>
//#include <iostream>
//#include <type_traits>
#include <stdlib.h>
//#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "ReluFPGA.h"

using namespace aocl_utils;

void cleanup()
{
	
}


class ReluFPGAImp
{
private:
	cl_platform_id platform_ = NULL;
	unsigned num_devices_ = 0;
	scoped_array<cl_device_id> device;
	cl_device_id target_device;
	// cl_device_id* device=NULL;
	cl_context context = NULL;
	// cl_command_queue queue; // num_devices elements
	cl_program program = NULL; 
	// scoped_array<cl_kernel> kernel; // num_devices elements
	// cl_kernel kernel;
	
	scoped_array<unsigned> rows_per_device; // num_devices elements


public:
	// explicit MatConvFPGAImp(unsigned A_height_, unsigned A_width_, unsigned B_height_, unsigned B_width_, unsigned C_height_, unsigned C_width_)
	// 	:  A_height(A_height_), A_width(A_width_), B_height(B_height_), B_width(B_width_), C_height(C_height_), C_width(C_width_) {}

	ReluFPGAImp() {
		#ifdef BREAKDOWN_TIME
		const double start_time11 = getCurrentTimestamp();
		#endif
		cl_int status;
		platform_ = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
		device.reset(getDevices(platform_, CL_DEVICE_TYPE_ALL, &num_devices_));
		target_device=device[0];
		context = clCreateContext(NULL, num_devices_, &target_device, &oclContextCallback, NULL, &status);
		std::string filepath="/home/jliu/hexin_workspace/fpga_tf/opencl_kernel_aocx/Relu_fpga.aocx";//+binary_file;
  		// printf("filepath: %s\n", filepath.c_str());
  		// printf("Using AOCX: %s\n", binary_file.c_str());
  		program = createProgramFromBinary(context, filepath.c_str(), &target_device, num_devices_);
  		// printf("\nfunction call\n");

  		// Build the program that was just created.
  		status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  		  		// const char *kernel_name = "matrixMult";
  // 		kernel.reset(num_devices_);
		// kernel[0] = clCreateKernel(program, "matrixMult", &status);

  		// kernel = clCreateKernel(program, "matrixMult", &status);

  		// queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);

  		#ifdef BREAKDOWN_TIME
	    const double end_time11 = getCurrentTimestamp();
  		const double total_time11 = end_time11 - start_time11;
  		printf("\ninitial opencl env Time: %0.3f ms\n", total_time11 * 1e3);
  		#endif
	}

    void Compute(const void *in)  {
    	unsigned input_rows=in_rows;
		unsigned input_cols=in_cols;
		unsigned filter_rows=filt_rows;
		unsigned filter_cols=filt_cols;
		#ifdef BREAKDOWN_TIME
		printf("input size: %d x %d\n", in_rows, in_cols);
		printf("filter size: %d x %d\n", filt_rows, filt_rows);
		return;
		#endif

		const float *input=(const float*)in;
		const float *filter=(const float*)filt;
		float *output=(float *)out;

		cl_int status;
		cl_kernel kernel;

		// bool flag=false;
		// if(A_height%BLOCK_SIZE==0&&A_width%BLOCK_SIZE==0&&B_width%BLOCK_SIZE==0) {
		if (BLOCK_SIZE>=256)
		{
			kernel = clCreateKernel(program, "relu", &status);
		}
		// else if (BLOCK_SIZE>=128)
		// {
		// 	kernel = clCreateKernel(program, "matrixMult128", &status);
		// }
		// else if(BLOCK_SIZE>=64)
		// {
		// 	kernel = clCreateKernel(program, "matrixMult64", &status);
		// }
		// 	flag=true;
		// }
		// else {
		// 	kernel = clCreateKernel(program, "matrixMult", &status);
		// }

		cl_command_queue queue; // num_devices elements
		// scoped_array<cl_kernel> kernel; // num_devices elements

		queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE, &status);
		
		cl_mem input_buf; // num_devices elements
		cl_mem filter_buf; // num_devices elements
		cl_mem output_buf; // num_devices elements

		input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_BANK_1_ALTERA*/, input_rows * input_cols * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input");

		filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_BANK_2_ALTERA*/, filter_rows * filter_cols * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for filter");

		output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY/* | CL_MEM_BANK_1_ALTERA*/, input_rows * input_cols * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for output");

	  // Transfer inputs to each device. Each of the host buffers supplied to
	  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
	  // for the host-to-device transfer.
	  	// for(unsigned i = 0; i < num_devices_; ++i) {
		#ifdef BREAKDOWN_TIME
		const double start_time1 = getCurrentTimestamp();
		#endif
		status = clEnqueueWriteBuffer(queue, input_buf, CL_FALSE,
		        0, input_rows * input_cols * sizeof(float), input, 0, NULL, NULL);
		    checkError(status, "Failed to transfer input");

		    status = clEnqueueWriteBuffer(queue, filter_buf, CL_FALSE,
		        0, filter_rows * filter_cols * sizeof(float), filter, 0, NULL, NULL);
		    checkError(status, "Failed to transfer filter");
 		// }

	  		// Wait for all queues to finish.
	  	// for(unsigned i = 0; i < num_devices_; ++i) {
	    	clFinish(queue);
	  	// }
	    #ifdef BREAKDOWN_TIME
	    const double end_time1 = getCurrentTimestamp();
  		const double total_time1 = end_time1 - start_time1;
  		printf("\ncpu->fpga Time: %0.3f ms\n", total_time1 * 1e3);
  		#endif
		  // Launch kernels.
		  // This is the portion of time that we'll be measuring for throughput
		  // benchmarking.
  		unsigned half_filter_size=filter_rows/2;
  		
		 scoped_array<cl_event> kernel_event(num_devices_);
		 #ifdef BREAKDOWN_TIME
		 const double start_time = getCurrentTimestamp();
		 #endif
	  	// for(unsigned i = 0; i < num_devices_; ++i) {
		    // Set kernel arguments.
		    unsigned argi = 0;
		 	// float bias=1;

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &filter_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(input_rows), &input_rows);
		    checkError(status, "Failed to set argument %d", argi - 1);

	  		status = clSetKernelArg(kernel, argi++, sizeof(half_filter_size), &(half_filter_size));
		    checkError(status, "Failed to set argument %d", argi - 1);

		size_t global[2];
		size_t local[2];
		// size_t wg_info[3]; 
		// clGetKernelWorkGroupInfo(kernel, target_device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wg_info), wg_info, NULL);
		global[0] = input_rows;
	 	global[1] = input_rows;
		local[0] = filter_rows;
		local[1] = filter_rows;
		    

		//    const size_t global_work_size[2] = {input_cols, input_rows};
		  //  const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};

		    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
		        (size_t*)&global, (size_t*)&local, 0, NULL, &kernel_event[0]);
		    checkError(status, "Failed to launch kernel");

		    clFinish(queue);
	  	// }

		  // Wait for all kernels to finish.
		// clWaitForEvents(num_devices_, kernel_event);
		#ifdef BREAKDOWN_TIME
  		const double end_time = getCurrentTimestamp();
  		const double total_time = end_time - start_time;

  		// Wall-clock time taken.
  		printf("\nTime: %0.3f ms\n", total_time * 1e3);
  		
  		// Get kernel times using the OpenCL event profiling API.
  		for(unsigned i = 0; i < num_devices_; ++i) {
    		cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    		printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  		}
  		#endif
  		// Compute the throughput (GFLOPS).
  		// There are C_width * C_height output values, with each value
  		// computed using A_width multiplies and adds.
  		// const float flops = (float)(2.0f * C_width * C_height * A_width / total_time);
  		// printf("\nThroughput: %0.2f GFLOPS\n\n", flops * 1e-9);
  // 		for (unsigned i = 0; i < C_height*C_width; ++i)
		// {
		// 	printf("output_mat=%f\n", output[0][i]);
		// }

  		// Release kernel events.
  		// for(unsigned i = 0; i < num_devices_; ++i) {
    		clReleaseEvent(kernel_event[0]); 
  		// }
    		// scoped_aligned_ptr<float> out_temp(C_height*C_width);
  		// Read the result.
  		// for(unsigned i = 0; i < num_devices_; ++i) {
  			#ifdef BREAKDOWN_TIME
  			const double start_time2 = getCurrentTimestamp();
  			#endif
    		status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        	0, input_rows * input_cols * sizeof(float), output, 0, NULL, NULL);
    		checkError(status, "Failed to read output matrix");
    		clFinish(queue);
    		#ifdef BREAKDOWN_TIME
    		const double end_time2 = getCurrentTimestamp();
    		const double total_time2 = end_time2 - start_time2;
    		printf("\nfpga->cpu Time: %0.3f ms\n", total_time2 * 1e3);
    		#endif

	  //   	if(kernel && kernel[0]) {
			//     clReleaseKernel(kernel[0]);
			// }
			// if(queue && queue[0]) {
			//     clReleaseCommandQueue(queue[0]);
			// }
			// if(kernel) {
			// 	clReleaseKernel(kernel);
			// }
    		if(kernel) {
				clReleaseKernel(kernel);
			}

			if(queue) {
				clReleaseCommandQueue(queue);
			}

		    if(input_buf) {
		      clReleaseMemObject(input_buf);
		    }
		    if(filter_buf) {
		      clReleaseMemObject(filter_buf);
		    }
		    if(output_buf) {
		      clReleaseMemObject(output_buf);
		    }

     }      //Implement function

	void CleanUp() {         //Clean OpenCL object

	  	if(program) {
	    	clReleaseProgram(program);
	  	}
	  	if(context) {
	    	clReleaseContext(context);
	  	}

	}
};

ReluFPGAImpInterface::ReluFPGAImpInterface() {
	_ReluFPGAImp = new ReluFPGAImp();
} 

ReluFPGAImpInterface::~ReluFPGAImpInterface() {
	#ifdef BREAKDOWN_TIME
	const double start_time4 = getCurrentTimestamp();
	#endif
	_ReluFPGAImp->CleanUp();
	#ifdef BREAKDOWN_TIME
	const double end_time4 = getCurrentTimestamp();
    const double total_time4 = end_time4 - start_time4;
	printf("\ncleanup Time: %0.3f ms\n", total_time4 * 1e3);
	#endif
} 

void ReluFPGAImpInterface::Compute(const void *in) {
	_ReluFPGAImp->Compute(in, filt, out, in_rows, in_cols, filt_rows, filt_cols);
} 


// int main(int argc, char const *argv[])
// {
// 	// MatConvFPGAInterface *matconv_fpga_;
// 	MatConvFPGAInterface *matconv_fpga_ = new MatConvFPGAInterface();
//     matconv_fpga_->InitOpencl();
//     matconv_fpga_->Compute();
//     matconv_fpga_->CleanUp();
// 	return 0;
// }
