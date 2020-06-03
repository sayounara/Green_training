#define BLOCK_SIZE 32
//#define FILTER_SIZE 3
__kernel 
__attribute((num_compute_units(2)))
__attribute((num_simd_work_items(8)))
__attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void conv_2d1(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
//    const float pBias,
      int input_size,
      int FILTER_SIZE)                // constant offset/bias
{
    __local float local_image[BLOCK_SIZE * BLOCK_SIZE];
    __local float local_filt[FILTER_SIZE * FILTER_SIZE];

        int x = get_local_id(0);
        int y = get_local_id(1);
        if(x < FILTER_SIZE*FILTER_SIZE) {
            local_filt[x] = filt[x];
        }
        local_image[y * input_size + x] = in[y * input_size + x];
    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
        int i = get_local_id(0);
        int j = get_local_id(1);
        //__attribute__((opencl_unroll_hint))
        for (int r = 0; r < FILTER_SIZE; r++) 
        {
            #pragma unroll
            for(int c = 0; c < FILTER_SIZE; c++)
            {
                sum += local_filt[r * FILTER_SIZE + c]*local_image[(j + r) * input_size + i + c];
            }
        }
        out[j * input_size + i] = sum; //+ pBias;
}

#define BLOCK_SIZE 32
//#define FILTER_SIZE 3
__kernel 
__attribute((num_compute_units(1)))
__attribute((num_simd_work_items(4)))
__attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void conv_2d2(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
//    const float pBias,
      int input_size,
      int FILTER_SIZE)                // constant offset/bias
{
    __local float local_image[BLOCK_SIZE * BLOCK_SIZE];
    __local float local_filt[FILTER_SIZE * FILTER_SIZE];

        int x = get_local_id(0);
        int y = get_local_id(1);
        if(x < FILTER_SIZE*FILTER_SIZE) {
            local_filt[x] = filt[x];
        }
        local_image[y * input_size + x] = in[y * input_size + x];
    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
        int i = get_local_id(0);
        int j = get_local_id(1);
        //__attribute__((opencl_unroll_hint))
        for (int r = 0; r < FILTER_SIZE; r++) 
        {
            #pragma unroll
            for(int c = 0; c < FILTER_SIZE; c++)
            {
                sum += local_filt[r * FILTER_SIZE + c]*local_image[(j + r) * input_size + i + c];
            }
        }
        out[j * input_size + i] = sum; //+ pBias;
}

#define BLOCK_SIZE 64
//#define FILTER_SIZE 3
__kernel 
__attribute((num_compute_units(2)))
__attribute((num_simd_work_items(16)))
__attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void conv_2d3(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
//    const float pBias,
      int input_size,
      int FILTER_SIZE)                // constant offset/bias
{
    __local float local_image[BLOCK_SIZE * BLOCK_SIZE];
    __local float local_filt[FILTER_SIZE * FILTER_SIZE];

        int x = get_local_id(0);
        int y = get_local_id(1);
        if(x < FILTER_SIZE*FILTER_SIZE) {
            local_filt[x] = filt[x];
        }
        local_image[y * input_size + x] = in[y * input_size + x];
    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
        int i = get_local_id(0);
        int j = get_local_id(1);
        //__attribute__((opencl_unroll_hint))
        for (int r = 0; r < FILTER_SIZE; r++) 
        {
            #pragma unroll
            for(int c = 0; c < FILTER_SIZE; c++)
            {
                sum += local_filt[r * FILTER_SIZE + c]*local_image[(j + r) * input_size + i + c];
            }
        }
        out[j * input_size + i] = sum; //+ pBias;
}


/**
//#define IMAGE_HEIGHT   (28)
//#define IMAGE_WIDTH    (28)
#define BLOCK_SIZE 7
#define FILTER_SIZE    (3)
//__kernel __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
__kernel __attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void conv_2d(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
    const float pBias,
    const int input_size)                // constant offset/bias
{
    __local float local_image[BLOCK_SIZE * BLOCK_SIZE];
    __local float local_filt[FILTER_SIZE * FILTER_SIZE];

//    __attribute__((xcl_pipeline_workitems)) {
        int x = get_local_id(0);
        int y = get_local_id(1);
        if(x < FILTER_SIZE*FILTER_SIZE) {
            local_filt[x] = filt[x];
        }
        local_image[y * input_size + x] = in[y * input_size + x];
  //  }
    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
  //  __attribute__((xcl_pipeline_workitems)) {
        int i = get_local_id(0);
        int j = get_local_id(1);
        //__attribute__((opencl_unroll_hint))
	#pragma unroll
        for (int r = 0; r < FILTER_SIZE; r++) 
        {
            #pragma unroll
            for(int c = 0; c < FILTER_SIZE; c++)
            {
                sum += local_filt[r * FILTER_SIZE + c]*local_image[(j + r) * input_size + i + c];
            }
        }
        out[j * input_size + i] = sum + pBias;
    //}
}




*/

