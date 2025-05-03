/*
  Simple matrix multiplication implementation: C = A × B

  This implementation uses a smaller block size for testing, but otherwise
  reuses the logic from test_var001.c with blocking support.

  - Modified for clarity and minimalism
*/

#include <stdio.h>
#include <stdlib.h>

#include "instruments.h"
#include "COMPUTE.h"

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef COMPUTE_MODEL_NAME
#define COMPUTE_MODEL_NAME baseline_model
#endif

#define BM 16  // Block size in M dimension
#define BN 16  // Block size in N dimension
#define BK 16  // Block size in K dimension

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

    // Count total floating-point operations and memory read/writes
    model->flops = 2.0 * m * n * k;
    model->bytes = sizeof(float) * (m * k + k * n + m * n); // Memory traffic: A, B, and C
}

// Actual matrix multiplication with 3-level loop blocking
void COMPUTE_NAME(op_params_t *op_params,
                  op_inputs_t *inputs,
                  op_outputs_t *outputs,
                  op_inouts_t *inouts,
                  hwctx_t *hwctx)
{
    int m = op_params->m;
    int n = op_params->n;
    int k = op_params->k;

    // Strides for row/column access of A, B, and C
    int rs_a = op_params->rs_a;
    int cs_a = op_params->cs_a;
    int rs_b = op_params->rs_b;
    int cs_b = op_params->cs_b;
    int rs_c = op_params->rs_c;
    int cs_c = op_params->cs_c;

    float *A = inputs->A;
    float *B = inputs->B;
    float *C = outputs->C;

    // Initialize C to zero before accumulation
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * rs_c + j * cs_c] = 0.0f;
        }
    }

    BEGIN_INSTRUMENTATION;

    // Outer loop: block across m-dimension
    for (int i0 = 0; i0 < m; i0 += BM)
    {
        // Middle loop: block across n-dimension
        for (int j0 = 0; j0 < n; j0 += BN)
        {
            // Inner loop: block across k-dimension
            for (int p0 = 0; p0 < k; p0 += BK)
            {
                // Compute valid block boundaries in each dimension
                int i_max = (i0 + BM > m) ? m : i0 + BM;
                int j_max = (j0 + BN > n) ? n : j0 + BN;
                int p_max = (p0 + BK > k) ? k : p0 + BK;

                // Perform computation over the current block
                for (int i = i0; i < i_max; ++i)
                {
                    for (int j = j0; j < j_max; ++j)
                    {
                        float sum = 0.0f;

                        // Compute dot product for C[i][j]
                        for (int p = p0; p < p_max; ++p)
                        {
                            float a_val = A[i * rs_a + p * cs_a];
                            float b_val = B[p * rs_b + j * cs_b];
                            sum += a_val * b_val;
                        }

                        int c_idx = i * rs_c + j * cs_c;
                        C[c_idx] += sum;  // Accumulate into output
                    }
                }
            }
        }
    }

    END_INSTRUMENTATION;
}
