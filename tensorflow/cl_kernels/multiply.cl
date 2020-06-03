__kernel
__attribute((num_compute_units(NUM_CU)))
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void multiply(global const float *a,
	global const float *b,
	global float *c)
{
	int id=get_global_id(0);
	c[id]=a[id] * b[id];
}
