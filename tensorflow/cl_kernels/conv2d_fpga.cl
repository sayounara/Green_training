 #define BLOCK_SIZE 1
 #define CU_NUM 2
 #define SIMD_NUM 8

__kernel 
__attribute((num_compute_units(CU_NUM)))
__attribute((num_simd_work_items(SIMD_NUM)))
__attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void convolute(
	const __global float * input, 
	__global float * output,
	__constant float * filter,
	int INPUT_SIZE,
	int HALF_FILTER_SIZE 
)
{

	int rowOffset = get_global_id(1) * INPUT_SIZE * 4;
	int my = 4 * get_global_id(0) + rowOffset;

	if (
		get_global_id(0) < HALF_FILTER_SIZE || 
		get_global_id(0) > INPUT_SIZE - HALF_FILTER_SIZE - 1 || 
		get_global_id(1) < HALF_FILTER_SIZE ||
		get_global_id(1) > INPUT_SIZE - HALF_FILTER_SIZE - 1
	)
	{
		/*
		output[my] = 0.0;
		output[my+1] = 255.0;
		output[my+2] = 255.0;
		output[my+3] = 255.0;
		*/
		
		return;
	}
	
	else
	{
		// perform convolution
		int fIndex = 0;
		output[my] = 0.0;
		output[my+1] = 0.0;
		output[my+2] = 0.0;
		output[my+3] = 0.0;
		
		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
		{
			int curRow = my + r * (INPUT_SIZE * 4);
			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
			{
				int offset = c * 4;
				
				output[ my   ] += input[ curRow + offset   ] * filter[ fIndex   ]; 
				output[ my+1 ] += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
				output[ my+2 ] += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
				output[ my+3 ] += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
				
				fIndex += 4;
	
			}
		}
	}
}
