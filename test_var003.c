/*
  Simple matrix multiplication implementation: C = A × B

  This implementation builds off test_var001.c and implements a micro-kernel

  - Modified for clarity and minimalism
*/

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> 

#include "instruments.h"
#include "COMPUTE.h"

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef COMPUTE_MODEL_NAME
#define COMPUTE_MODEL_NAME baseline_model
#endif

#define BM 64
#define BN 64
#define BK 64

// Performance model: estimate FLOPs and memory traffic
void COMPUTE_MODEL_NAME(op_model_t *model,
                        op_params_t *op_params,
                        op_inputs_t *inputs,
                        op_outputs_t *outputs,
                        op_inouts_t *inouts,
                        hwctx_t *hwctx)
{
    int m = op_params->m;
    int n = op_params->n;
    int k = op_params->k;

    model->flops = 2.0 * m * n * k;
    model->bytes = sizeof(float) * (m * k + k * n + m * n); // A, B, and C
}

// Micro-kernel: performs 1x8 matrix multiplication using AVX
void microkernel_1x8(int k,
    const float *A,  // Pointer to row of A (A[i][p])
    const float *B,  // Pointer to row panel of B (B[p][j])
    float *C,        // Pointer to row of C (C[i][j])
    int rs_b,        // Row stride for B
    int cs_b,        // Column stride for B
    int rs_c,        // Row stride for C
    int cs_c)        // Column stride for C
{
    // Load current values from C[i][j..j+7] into SIMD register
    __m256 c_vec = _mm256_loadu_ps(C);

    // Loop over k dimension (depth)
    for (int p = 0; p < k; ++p) {
        float a_val = A[p];  // A[i][p] scalar

        // Manually gather B[p][j..j+7] into temporary array
        float b_temp[8];
        for (int t = 0; t < 8; ++t)
            b_temp[t] = B[p * rs_b + t * cs_b];

        // Load B[p][j..j+7] into SIMD register
        __m256 b_vec = _mm256_loadu_ps(b_temp);

        // Broadcast A[i][p] across SIMD lanes and perform fused multiply-add
        __m256 a_broadcast = _mm256_set1_ps(a_val);
        c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
    }

    // Store updated results back into C[i][j..j+7]
    _mm256_storeu_ps(C, c_vec);
}

// Main matrix multiplication function
void COMPUTE_NAME(op_params_t *op_params,
                  op_inputs_t *inputs,
                  op_outputs_t *outputs,
                  op_inouts_t *inouts,
                  hwctx_t *hwctx)
{
    int m = op_params->m;
    int n = op_params->n;
    int k = op_params->k;

    int rs_a = op_params->rs_a;
    int cs_a = op_params->cs_a;
    int rs_b = op_params->rs_b;
    int cs_b = op_params->cs_b;
    int rs_c = op_params->rs_c;
    int cs_c = op_params->cs_c;

    float *A = inputs->A;
    float *B = inputs->B;
    float *C = outputs->C;

    // Zero initialize the output matrix C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * rs_c + j * cs_c] = 0.0f;
        }
    }

    BEGIN_INSTRUMENTATION;

    // Loop over blocks of A, B, and C
    for (int i0 = 0; i0 < m; i0 += BM) {
        for (int j0 = 0; j0 < n; j0 += BN) {
            for (int p0 = 0; p0 < k; p0 += BK) {

                int i_max = (i0 + BM > m) ? m : i0 + BM;
                int j_max = (j0 + BN > n) ? n : j0 + BN;
                int p_max = (p0 + BK > k) ? k : p0 + BK;

                // Loop over rows in the current block of A/C
                for (int i = i0; i < i_max; ++i) {
                    // Loop over columns in blocks of 8 for SIMD micro-kernel
                    for (int j = j0; j <= j_max - 8; j += 8) {
                        const float *A_row = &A[i * rs_a + p0 * cs_a];           // A[i][p0]
                        const float *B_panel = &B[p0 * rs_b + j * cs_b];        // B[p0][j]
                        float *C_row = &C[i * rs_c + j * cs_c];                 // C[i][j]

                        // Compute using AVX micro-kernel
                        microkernel_1x8(p_max - p0,
                            A_row,
                            B_panel,
                            C_row,
                            rs_b, cs_b,
                            rs_c, cs_c);
                    }

                    // Scalar fallback for columns not divisible by 8
                    for (int j = j_max - (j_max % 8); j < j_max; ++j) {
                        float sum = 0.0f;
                        for (int p = p0; p < p_max; ++p) {
                            float a_val = A[i * rs_a + p * cs_a];
                            float b_val = B[p * rs_b + j * cs_b];
                            sum += a_val * b_val;
                        }
                        C[i * rs_c + j * cs_c] += sum;
                    }
                }
            }
        }
    }

    END_INSTRUMENTATION;
}
