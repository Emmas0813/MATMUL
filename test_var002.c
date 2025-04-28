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

// Actual matrix multiplication

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

    // Zero initialize C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * rs_c + j * cs_c] = 0.0f;
        }
    }

    BEGIN_INSTRUMENTATION;

    for (int i0 = 0; i0 < m; i0 += BM)
{
    for (int j0 = 0; j0 < n; j0 += BN)
    {
        for (int p0 = 0; p0 < k; p0 += BK)
        {
            int i_max = (i0 + BM > m) ? m : i0 + BM;
            int j_max = (j0 + BN > n) ? n : j0 + BN;
            int p_max = (p0 + BK > k) ? k : p0 + BK;

            for (int i = i0; i < i_max; ++i) {
                for (int j = j0; j < j_max; ++j) {
                    __m256 vsum = _mm256_setzero_ps();
            
                    int p = p0;
                    // Vectorized part
                    for (; p <= p_max - 8; p += 8) {
                        // A: contiguous, safe for loadu
                        __m256 va = _mm256_loadu_ps(&A[i * rs_a + p * cs_a]);
                        
                        // B: not contiguous in memory, so pack 8 elements
                        float b_buf[8];
                        for (int t = 0; t < 8; ++t)
                            b_buf[t] = B[(p + t) * rs_b + j * cs_b];
                        __m256 vb = _mm256_loadu_ps(b_buf);
            
                        // FMA if available, else mul+add
                        vsum = _mm256_fmadd_ps(va, vb, vsum);
                    }
            
                    // Horizontal sum to get the float result
                    float sum = 0.0f;
                    float temp[8];
                    _mm256_storeu_ps(temp, vsum);
                    for (int t = 0; t < 8; ++t)
                        sum += temp[t];
            
                    // Scalar tail for any leftover p
                    for (; p < p_max; ++p) {
                        float a_val = A[i * rs_a + p * cs_a];
                        float b_val = B[p * rs_b + j * cs_b];
                        sum += a_val * b_val;
                    }
            
                    // Accumulate into C
                    int c_idx = i * rs_c + j * cs_c;
                    C[c_idx] += sum;
                }
            }

        }
    }
}


    END_INSTRUMENTATION;
}
