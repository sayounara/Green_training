//maxPool2d 
//kernel_size=3 stride=2
//output one feature map per kernel
__kernel void maxpool2d(
	const int input_size,
	const int output_size,
	__global float *input_im,
    __global float *restrict output_im)
{
	int channels = get_global_id(0);//get output channel index
	
	input_im += channels * input_size * input_size;
	output_im += channels * output_size * output_size;

	//loop over output feature map
	for(int i = 0; i < output_size; i++)//row
	{
		for(int j = 0; j < output_size; j++)//col
		{
			//find the max value in 3x3 reigon 
			//to be one element in the output feature map
			float tmp = 0.0;

			#pragma unroll 1
			for(int k = 0; k < 3; k++)//row
			{
				#pragma unroll 1
				for(int l = 0; l < 3; l++)//col
				{
					float value = input_im[(i * 2 + k) * input_size  + j * 2 + l ];
					if(value > tmp)
						tmp = value;
				}
			}
			//store the result to output feature map
			output_im[i * output_size + j] = tmp; 
		}
	}
}
