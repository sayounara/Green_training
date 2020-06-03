__kernel void batch_norm_layer(
	__global float *pMaps,
	__global float *pScale,
	__global float *pOffset,
	__global float *pOutput) {
	
	const int map_no = get_global_id(2);
	const int row_no = get_global_id(1);
	const int col_no = get_global_id(0);
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	float norm_pix;
	const int pos = map_no*W*H + row_no*W + col_no;
	norm_pix = pMaps[pos] * pScale[map_no] + pOffset[map_no];
	pOutput[pos] = norm_pix;
}
