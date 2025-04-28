/*
  Simple matrix multiplication implementation: C = A × B

  This implementation uses a larger size block size but uses test_var001.c code otherwise

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

#define BM 16
#define BN 16
#define BK 16

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

            for (int i = i0; i < i_max; ++i)
            {
                for (int j = j0; j < j_max; ++j)
                {
                    float sum = 0.0f;
                    for (int p = p0; p < p_max; ++p)
                    {
                        float a_val = A[i * rs_a + p * cs_a];
                        float b_val = B[p * rs_b + j * cs_b];
                        sum += a_val * b_val;
                    }

                    int c_idx = i * rs_c + j * cs_c;
                  
                    C[c_idx] += sum;
                }
            }
        }
    }
}


    END_INSTRUMENTATION;
}
