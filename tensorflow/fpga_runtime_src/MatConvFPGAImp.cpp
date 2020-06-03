#include <stdio.h>
//#include <iostream>
//#include <type_traits>
#include <stdlib.h>
//#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "MatConvFPGAImp.h"

using namespace aocl_utils;

void cleanup()
{
	
}


class MatConvFPGAImp
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

	MatConvFPGAImp() {
		#ifdef BREAKDOWN_TIME
		const double start_time11 = getCurrentTimestamp();
		#endif
		cl_int status;
		platform_ = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
		device.reset(getDevices(platform_, CL_DEVICE_TYPE_ALL, &num_devices_));
		target_device=device[0];
		context = clCreateContext(NULL, num_devices_, &target_device, &oclContextCallback, NULL, &status);
		std::string filepath="../fpga_bitsreams/matrix_mult.aocx";//+binary_file;
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

    void Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_)  {
	    unsigned A_height = A_height_;
		unsigned A_width  = A_width_;
		const unsigned &B_height = A_width;
		unsigned B_width  = B_width_;
		const unsigned &C_height = A_height;
		// unsigned C_height = A_height;
		const unsigned &C_width  = B_width;
		// unsigned C_width  = B_width;
		const float *input_a=(const float *)matrix_a;
		const float *input_b=(const float *)matrix_b;
		#ifdef BREAKDOWN_TIME
		printf("Matrix A: %d x %d\n", A_height, A_width);
		printf("Matrix B: %d x %d\n", B_height, B_width);
		#endif
		// for (int i = 0; i < 10; ++i)
		// {
		// 	printf("%f ", input_a[i]);
		// }
		float *output=(float *)matrix_out;

		cl_int status;
		cl_kernel kernel;


		// bool flag=false;
		// if(A_height%BLOCK_SIZE==0&&A_width%BLOCK_SIZE==0&&B_width%BLOCK_SIZE==0) {
		bool isDivisible=false;
		if(A_height%64==0 && B_width%64==0) {
			isDivisible=true;
			if (BLOCK_SIZE>=256)
			{
				kernel = clCreateKernel(program, "matrixMult256", &status);
			}
			else if (BLOCK_SIZE>=128)
			{
				kernel = clCreateKernel(program, "matrixMult128", &status);
			}
			else if(BLOCK_SIZE>=64)
			{
				kernel = clCreateKernel(program, "matrixMult64", &status);
			}
		}
		else
		{
			kernel = clCreateKernel(program, "matrixMult256", &status);
			// if(A_width==1001)
			// {
			// 	kernel = clCreateKernel(program, "matrixMult256", &status);
			// }
			// else
			// 	kernel = clCreateKernel(program, "matrixMult_padding", &status);
			// printf("Indivisble Matrix A and B~~\n");
		}
		// 	flag=true;
		// }
		// else {
		// 	kernel = clCreateKernel(program, "matrixMult", &status);
		// }

		cl_command_queue queue; // num_devices elements
		// scoped_array<cl_kernel> kernel; // num_devices elements

		queue = clCreateCommandQueue(context, target_device, CL_QUEUE_PROFILING_ENABLE/* | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/, &status);//device does not support CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
		
		cl_mem input_a_buf; // num_devices elements
		cl_mem input_b_buf; // num_devices elements
		cl_mem output_buf; // num_devices elements

  		input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_CHANNEL_2_INTELFPGA*/, 
		        A_height * A_width * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input A");

		    // For matrix B, each device needs the whole matrix. We specifically
		    // assign this buffer to the second bank of global memory.
		input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_BANK_2_ALTERA*/, 
		        B_height * B_width * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for input B");

		    // Output buffer. This is matrix C, for the rows that are computed by this
		    // device. We assign this buffer to the first bank of global memory,
		    // although it is not material to performance to do so because
		    // the reads from the input matrices are far more frequent than the
		    // write to the output matrix.
		output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY/* | CL_MEM_BANK_1_ALTERA*/, 
		        C_height * C_width * sizeof(float), NULL, &status);
		checkError(status, "Failed to create buffer for output");

	  // Transfer inputs to each device. Each of the host buffers supplied to
	  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
	  // for the host-to-device transfer.
	  	// for(unsigned i = 0; i < num_devices_; ++i) {
		#ifdef BREAKDOWN_TIME
		const double start_time1 = getCurrentTimestamp();
		#endif
		    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
		        0, A_height * A_width * sizeof(float), input_a, 0, NULL, NULL);
		    checkError(status, "Failed to transfer input A");

		    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
		        0, B_width * B_height * sizeof(float), input_b, 0, NULL, NULL);
		    checkError(status, "Failed to transfer input B");
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
		 scoped_array<cl_event> kernel_event(num_devices_);
		 #ifdef BREAKDOWN_TIME
		 const double start_time = getCurrentTimestamp();
		 #endif
	  	// for(unsigned i = 0; i < num_devices_; ++i) {
		    // Set kernel arguments.
		    unsigned argi = 0;

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(A_width), &A_width);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    status = clSetKernelArg(kernel, argi++, sizeof(B_width), &B_width);
		    checkError(status, "Failed to set argument %d", argi - 1);

		    // Enqueue kernel.
		    // Use a global work size corresponding to the size of the output matrix.
		    // Each work-item computes the result for one value of the output matrix,
		    // so the global work size has the same dimensions as the output matrix.
		    // 
		    // The local work size is one block, so BLOCK_SIZE x BLOCK_SIZE.
		    //
		    // Events are used to ensure that the kernel is not launched until
		    // the writes to the input buffers have completed.
		    // const size_t global_work_size[2] = {C_width, C_height};
		    // const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};
		    // printf("Launching for device %d (global size: %d, %d)\n", i, global_work_size[0], global_work_size[1]);
		    if(isDivisible)
		    {
		    	const size_t global_work_size[2] = {C_width, C_height};
				const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};

		    	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event[0]);
			}
			else
			{
				// const size_t global_work_size[2] = {C_width, C_height};
				const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};
				// const size_t local_work_size[2]  = {64, 64};
				if(A_height%BLOCK_SIZE!=0)
				{
					size_t temp=A_height/BLOCK_SIZE+1;
					const size_t global_work_size[2] = {C_width, temp*BLOCK_SIZE};
					// A_height=1024;
					status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event[0]);
				}
				else if(B_width%BLOCK_SIZE!=0)
				{
					size_t temp=B_width/BLOCK_SIZE+1;
					const size_t global_work_size[2] = {temp*BLOCK_SIZE, C_height};
					// B_width=1024;
					status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event[0]);
				}
				// else //if(A_width==1001)
				// {
				// 	const size_t global_work_size[2] = {C_width, C_height};
				// 	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event[0]);
				// }
				// const size_t global_work_size[2] = {C_width, C_height};
				// status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernel_event[0]);
				
			}
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
    		status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
        	0, /*rows_per_device[0]*/ C_height * C_width * sizeof(float), output, 0, NULL, NULL);
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

		    if(input_a_buf) {
		      clReleaseMemObject(input_a_buf);
		    }
		    if(input_b_buf) {
		      clReleaseMemObject(input_b_buf);
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

MatConvFPGAImpInterface::MatConvFPGAImpInterface() {
	_MatConvFPGAImp = new MatConvFPGAImp();
} 

MatConvFPGAImpInterface::~MatConvFPGAImpInterface() {
	#ifdef BREAKDOWN_TIME
	const double start_time4 = getCurrentTimestamp();
	#endif
	_MatConvFPGAImp->CleanUp();
	#ifdef BREAKDOWN_TIME
	const double end_time4 = getCurrentTimestamp();
    const double total_time4 = end_time4 - start_time4;
	printf("\ncleanup Time: %0.3f ms\n", total_time4 * 1e3);
	#endif
} 

void MatConvFPGAImpInterface::Compute(const void *matrix_a, const void *matrix_b, void *matrix_out, unsigned A_height_ ,unsigned A_width_, unsigned B_width_) {
	_MatConvFPGAImp->Compute(matrix_a, matrix_b, matrix_out, A_height_, A_width_, B_width_);
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
