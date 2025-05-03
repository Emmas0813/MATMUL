/*
  Simple matrix multiplication implementation: C = A × B

  This implementation builds off test_var001.c and implements SIMD vectorization

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

#define BM 64  // Block size in M dimension
#define BN 64  // Block size in N dimension
#define BK 64  // Block size in K dimension

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

    // Count floating-point operations and memory reads/writes
    model->flops = 2.0 * m * n * k;
    model->bytes = sizeof(float) * (m * k + k * n + m * n); // Total memory traffic for A, B, and C
}

// Actual matrix multiplication with SIMD vectorization
void COMPUTE_NAME(op_params_t *op_params,
                  op_inputs_t *inputs,
                  op_outputs_t *outputs,
                  op_inouts_t *inouts,
                  hwctx_t *hwctx)
{
    int m = op_params->m;
    int n = op_params->n;
    int k = op_params->k;

    // Row and column strides for A, B, and C
    int rs_a = op_params->rs_a;
    int cs_a = op_params->cs_a;
    int rs_b = op_params->rs_b;
    int cs_b = op_params->cs_b;
    int rs_c = op_params->rs_c;
    int cs_c = op_params->cs_c;

    float *A = inputs->A;
    float *B = inputs->B;
    float *C = outputs->C;

    // Zero initialize C matrix
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * rs_c + j * cs_c] = 0.0f;
        }
    }

    BEGIN_INSTRUMENTATION;

    // Blocked loop structure for matrix multiplication
    for (int i0 = 0; i0 < m; i0 += BM) {
        for (int j0 = 0; j0 < n; j0 += BN) {
            for (int p0 = 0; p0 < k; p0 += BK) {

                int i_max = (i0 + BM > m) ? m : i0 + BM;
                int j_max = (j0 + BN > n) ? n : j0 + BN;
                int p_max = (p0 + BK > k) ? k : p0 + BK;

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {

                        __m256 vsum = _mm256_setzero_ps(); // Accumulator for 8-wide vector sum

                        int p = p0;
                        // Vectorized inner loop
                        for (; p <= p_max - 8; p += 8) {
                            // Load 8 A values (assumed to be contiguous)
                            __m256 va = _mm256_loadu_ps(&A[i * rs_a + p * cs_a]);

                            // Manually gather 8 B values since B is not laid out contiguously
                            float b_buf[8];
                            for (int t = 0; t < 8; ++t)
                                b_buf[t] = B[(p + t) * rs_b + j * cs_b];
                            __m256 vb = _mm256_loadu_ps(b_buf);

                            // Perform fused multiply-add: vsum += va * vb
                            vsum = _mm256_fmadd_ps(va, vb, vsum);
                        }

                        // Horizontal reduction: sum all 8 lanes into a scalar
                        float sum = 0.0f;
                        float temp[8];
                        _mm256_storeu_ps(temp, vsum);
                        for (int t = 0; t < 8; ++t)
                            sum += temp[t];

                        // Scalar remainder loop for leftover p values
                        for (; p < p_max; ++p) {
                            float a_val = A[i * rs_a + p * cs_a];
                            float b_val = B[p * rs_b + j * cs_b];
                            sum += a_val * b_val;
                        }

                        // Store final result into C[i][j]
                        int c_idx = i * rs_c + j * cs_c;
                        C[c_idx] += sum;
                    }
                }
            }
        }
    }

    END_INSTRUMENTATION;
}
