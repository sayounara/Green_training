__kernel void relu (__global float * pData)
{
        const int x = get_global_id(0);
        float zero = 0.0;
        pData[x] = fmax(zero,pData[x]);
}
