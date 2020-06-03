/*__kernel void winograd_2x2_3x3_16x16(
    __global float *inputs,
    __global float *outputs,
    __global float *filters,
    __global float *bias,
    int N,
    int C, int H, int W,
    int K, int P, int Q,
    int pad,
    int TP, int TQ, int BN, int BK,
    int TPmask, int TPwidth, int TPshift,
    int TQmask, int TQwidth, int TQshift,
    int Nmask, int Nwidth
    ) {
    int tptqbnbk = get_group_id(0);
    int tp = tptqbnbk / (TQ * BN * BK);
    int tqbnbk = tptqbnbk - tp * (TQ * BN * BK);
    int tq = tqbnbk / (BN * BK);
    int bnbk = tqbnbk - tq * (BN * BK);
    int bn = bnbk / (BK);
    int bk = bnbk - bn * (BK);

    int tid = get_local_id(0);
    int tidlow = tid & 15;
    int c = (tid & 0x70) >> 4;
    int ci = c - (C & 7 ? 8 - (C & 7) : 0);
    tp = (tp << TPwidth) + ((tid & TPmask) >> TPshift);
    tq = (tq << TQwidth) + ((tid & TQmask) >> TQshift);
    int h = (tp << 1) - pad, w = (tq << 1) - pad;
    int n = ((get_group_id(2) * BN + bn) << Nwidth) + (tid & Nmask);
    int k = ((get_group_id(1) * BK + bk) << 4) + tidlow;

    __local float SM[2 * 8 * 16 * 16];
    __local float *pRSV = SM + (tid & 0xf0) + (tid & 0x3);
    __local float *pRSU = SM + 8 * 16 * 16 + (tid & 0xf0) + ((tid & 0xc) >> 2);

    float r[4][4], rA[4], rB[4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r[i][j] = 0;
        }
    }

    if (tid < 128) { // image transform
        float v[4][4], TV[4][4], V[4][4];

        bool preds[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                preds[i][j] = n < N && 0 <= h + i && h + i < H && 0 <= w + j && w + j < W;
            }
        }

        __global float *pV = inputs + ((ci * H + h) * W + w) * N + n;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v[i][j] = ci >= 0 && preds[i][j] ? pV[(i * W + j) * N] : 0;
            }
        }

        __local float *pWSV = SM + c * 16 * 16 + tidlow;
        while (true) {
            TV[0][0] = v[0][0] - v[2][0];
            TV[0][1] = v[0][1] - v[2][1];
            TV[0][2] = v[0][2] - v[2][2];
            TV[0][3] = v[0][3] - v[2][3];

            TV[3][0] = v[1][0] - v[3][0];
            TV[3][1] = v[1][1] - v[3][1];
            TV[3][2] = v[1][2] - v[3][2];
            TV[3][3] = v[1][3] - v[3][3];

            TV[1][0] = v[1][0] + v[2][0];
            TV[1][1] = v[1][1] + v[2][1];
            TV[1][2] = v[1][2] + v[2][2];
            TV[1][3] = v[1][3] + v[2][3];

            TV[2][0] = v[2][0] - v[1][0];
            TV[2][1] = v[2][1] - v[1][1];
            TV[2][2] = v[2][2] - v[1][2];
            TV[2][3] = v[2][3] - v[1][3];

            V[0][0] = TV[0][0] - TV[0][2];
            V[0][3] = TV[0][1] - TV[0][3];
            V[3][0] = TV[3][0] - TV[3][2];
            V[3][3] = TV[3][1] - TV[3][3];

            V[1][0] = TV[1][0] - TV[1][2];
            V[2][0] = TV[2][0] - TV[2][2];
            V[1][3] = TV[1][1] - TV[1][3];
            V[2][3] = TV[2][1] - TV[2][3];

            V[2][1] = TV[2][1] + TV[2][2];
            V[2][2] = TV[2][2] - TV[2][1];

            V[0][1] = TV[0][1] + TV[0][2];
            V[0][2] = TV[0][2] - TV[0][1];
            V[1][1] = TV[1][1] + TV[1][2];
            V[1][2] = TV[1][2] - TV[1][1];
            V[3][1] = TV[3][1] + TV[3][2];
            V[3][2] = TV[3][2] - TV[3][1];

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSV[(i * 4 + j) * 16] = V[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 8; ++l) {
                for (int i = 0; i < 4; ++i) {
                    rA[i] = pRSU[l * 16 * 16 + i * 4];
                    rB[i] = pRSV[l * 16 * 16 + i * 4];
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 8;
            if (ci >= C) break;
            pV += 8 * H * W * N;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    v[i][j] = preds[i][j] ? pV[(i * W + j) * N] : 0;
                }
            }
        }
    } else { // filter transform
        float u[3][3], TU[4][3], TA[3], TB[4], U[4][4];

        bool pred = k < K;

        __global float *pU = filters + ci * 3 * 3 * K + k;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                u[i][j] = ci >= 0 && pred ? pU[(i * 3 + j) * K] : 0;
            }
        }

        __local float *pWSU = SM + (c + 8) * 16 * 16 + tidlow;
        while (true) {
            TA[0] = (u[0][0] + u[2][0]) * 0.5;
            TA[1] = (u[0][1] + u[2][1]) * 0.5;
            TA[2] = (u[0][2] + u[2][2]) * 0.5;
            TU[0][0] = u[0][0];
            TU[0][1] = u[0][1];
            TU[0][2] = u[0][2];
            TU[3][0] = u[2][0];
            TU[3][1] = u[2][1];
            TU[3][2] = u[2][2];
            TU[1][0] = TA[0] + u[1][0] * 0.5;
            TU[2][0] = TA[0] - u[1][0] * 0.5;
            TU[1][1] = TA[1] + u[1][1] * 0.5;
            TU[2][1] = TA[1] - u[1][1] * 0.5;
            TU[1][2] = TA[2] + u[1][2] * 0.5;
            TU[2][2] = TA[2] - u[1][2] * 0.5;
            TB[0] = (TU[0][0] + TU[0][2]) * 0.5;
            TB[1] = (TU[1][0] + TU[1][2]) * 0.5;
            TB[2] = (TU[2][0] + TU[2][2]) * 0.5;
            TB[3] = (TU[3][0] + TU[3][2]) * 0.5;
            U[0][0] = TU[0][0];
            U[0][3] = TU[0][2];
            U[3][0] = TU[3][0];
            U[3][3] = TU[3][2];
            U[1][0] = TU[1][0];
            U[2][0] = TU[2][0];
            U[1][3] = TU[1][2];
            U[2][3] = TU[2][2];
            U[1][1] = TB[1] + TU[1][1] * 0.5;
            U[1][2] = TB[1] - TU[1][1] * 0.5;
            U[2][1] = TB[2] + TU[2][1] * 0.5;
            U[2][2] = TB[2] - TU[2][1] * 0.5;
            U[0][1] = TB[0] + TU[0][1] * 0.5;
            U[0][2] = TB[0] - TU[0][1] * 0.5;
            U[3][1] = TB[3] + TU[3][1] * 0.5;
            U[3][2] = TB[3] - TU[3][1] * 0.5;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pWSU[(i * 4 + j) * 16] = U[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int l = 0; l < 8; ++l) {
                for (int i = 0; i < 4; ++i) {
                    rA[i] = pRSU[l * 16 * 16 + i * 4];
                    rB[i] = pRSV[l * 16 * 16 + i * 4];
                }
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        r[i][j] += rA[i] * rB[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            ci += 8;
            if (ci >= C) break;
            pU += 8 * 3 * 3 * K;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    u[i][j] = pred ? pU[(i * 3 + j) * K] : 0;
                }
            }
        }
    }

    // inverse transform
    {
        // log(16 * 16) - 2, log(16) - 4
        __local float *pWSM = SM + ((tid & 0x0c) << 6) + ((tid & 0xf0) << 0) + (tid & 0x03);
        __local float *pRSM = SM + ((tid & 0xf0) << 4) + tidlow;
        int oh = h + pad, ow = w + pad, on = n;
        int ok = k - tidlow + ((tid & 0xf0) >> 4);
        __global float *pO = outputs + ((ok * P + oh) * Q + ow) * N + on;

        bool preds[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                preds[i][j] = on < N && 0 <= oh + i && oh + i < P && 0 <= ow + j && ow + j < Q;
            }
        }

        {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    // log(4 * 16 * 16)
                    pWSM[(i << 10) + (j << 2)] = r[i][j];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            float m[4][4], TM[4][2], M[2][2];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    m[i][j] = pRSM[(i * 4 + j) * 16];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            TM[0][0] = m[0][0] + m[0][1] + m[0][2];
            TM[0][1] = m[0][1] - m[0][2] - m[0][3];
            TM[1][0] = m[1][0] + m[1][1] + m[1][2];
            TM[1][1] = m[1][1] - m[1][2] - m[1][3];
            TM[2][0] = m[2][0] + m[2][1] + m[2][2];
            TM[2][1] = m[2][1] - m[2][2] - m[2][3];
            TM[3][0] = m[3][0] + m[3][1] + m[3][2];
            TM[3][1] = m[3][1] - m[3][2] - m[3][3];

            M[0][0] = TM[0][0] + TM[1][0] + TM[2][0];
            M[0][1] = TM[0][1] + TM[1][1] + TM[2][1];
            M[1][0] = TM[1][0] - TM[2][0] - TM[3][0];
            M[1][1] = TM[1][1] - TM[2][1] - TM[3][1];

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    if (ok < K && preds[i][j]) {
                        pO[(i * Q + j) * N] = M[i][j] + bias[ok];
                    }
                }
            }
        }
    }
}
*/

/*__kernel void convolute(
    const __global float4 * input, 
    __global float4 * output,
    __constant float4 * filter,
    __local float4 * cached
)
{
    const int rowOffset = get_global_id(1) * IMAGE_W;
    const int my = get_global_id(0) + rowOffset;
    
    const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);
    const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen;
    const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;        
        
    // copy my pixel
    cached[ myLocal ] = input[ my ];

    
    if (
        get_global_id(0) < HALF_FILTER_SIZE             || 
        get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1       || 
        get_global_id(1) < HALF_FILTER_SIZE         ||
        get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
    )
    {
        // no computation for me, sync and exit
        barrier(CLK_LOCAL_MEM_FENCE);
        return;
    }
    else 
    {
        // copy additional elements
        int localColOffset = -1;
        int globalColOffset = -1;
        
        if ( get_local_id(0) < HALF_FILTER_SIZE )
        {
            localColOffset = get_local_id(0);
            globalColOffset = -HALF_FILTER_SIZE;
            
            cached[ localRowOffset + get_local_id(0) ] = input[ my - HALF_FILTER_SIZE ];
        }
        else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
        {
            localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
            globalColOffset = HALF_FILTER_SIZE;
            
            cached[ myLocal + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE ];
        }
        
        
        if ( get_local_id(1) < HALF_FILTER_SIZE )
        {
            cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE_IMAGE_W ];
            if (localColOffset > 0)
            {
                cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
            }
        }
        else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
        {
            int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
            cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE_IMAGE_W ];
            if (localColOffset > 0)
            {
                cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
            }
        }
        
        // sync
        barrier(CLK_LOCAL_MEM_FENCE);

        
        // perform convolution
        int fIndex = 0;
        float4 sum = (float4) 0.0;
        
        for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
        {
            int curRow = r * localRowLen;
            for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)
            {   
                sum += cached[ myLocal + curRow + c ] * filter[ fIndex ]; 
            }
        }
        output[my] = sum;
    }
}
*/

/**
aoc device/testconv.cl -o bin/temp/conv2d.aocx -board=s10_gh1e1_4Gx2 -v -report -fp-relaxed -no-interleaving=default -dont-error-if-large-area-est -D IMAGE_W=512 -D IMAGE_H=512 -D FILTER_SIZE=5 -D HALF_FILTER_SIZE=2 -D TWICE_HALF_FILTER_SIZE=4 -D HALF_FILTER_SIZE_IMAGE_W=1024 -I $INTELFPGAOCLSDKROOT/include/kernel_headers
*/

#include "ihc_apint.h"
//#define ap_int<8> int8_t
#define BLOCK_SIZE 256
__kernel 
__attribute((num_compute_units(8)))
__attribute((num_simd_work_items(16)))
__attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void convolute(
    const __global int8_t *restrict input, 
    __global int8_t *restrict output,
    __global int8_t *restrict filter 
)
{

    int rowOffset = get_global_id(1) * IMAGE_W * 4;
    int my = 4 * get_global_id(0) + rowOffset;
    
    int fIndex = 0;
    int8_t sumR = 0.0;
    int8_t sumG = 0.0;
    int8_t sumB = 0.0;
    int8_t sumA = 0.0;
    
        
    for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
    {
        int curRow = my + r * (IMAGE_W * 4);
        for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex += 4)
        {
            int offset = c * 4;
                
            sumR += input[ curRow + offset   ] * filter[ fIndex   ]; 
            sumG += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
            sumB += input[ curRow + offset+2 ] * filter[ fIndex+2 ]; 
            sumA += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
        }
    }
    
    output[ my     ] = sumR;
    output[ my + 1 ] = sumG;
    output[ my + 2 ] = sumB;
    output[ my + 3 ] = sumA;
    
}

