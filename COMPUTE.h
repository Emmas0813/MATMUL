#ifndef _COMPUTE_H
#define _COMPUTE_H

#include <stdio.h>

/* Operation-specific parameters for matrix multiplication */
typedef struct op_params_ts {
  int m;      // Rows of A and C
  int n;      // Columns of B and C
  int k;      // Columns of A, rows of B

  int rs_a;   // Row stride for A
  int cs_a;   // Column stride for A

  int rs_b;   // Row stride for B
  int cs_b;   // Column stride for B

  int rs_c;   // Row stride for C
  int cs_c;   // Column stride for C

  int m0;     // Optional tuning param (e.g. for benchmark naming)
  int n0;     // Optional tuning param (e.g. for benchmark naming)
} op_params_t;

/* Inputs to the matrix multiplication */
typedef struct op_inputs_ts {
  float *A;
  float *B;
} op_inputs_t;

/* Output of the matrix multiplication */
typedef struct op_outputs_ts {
  float *C;
} op_outputs_t;

/* Placeholder for in-place operations (unused in this case) */
typedef struct op_inouts_ts {
  float *inout;
} op_inouts_t;

/* Performance model: flop count and memory traffic in bytes */
typedef struct op_model_ts {
  float flops;
  float bytes;
} op_model_t;

/* Hardware context (extendable later) */
typedef struct hwctx_ts {
  int total_available_threads;
} hwctx_t;

/* Benchmark configuration */
typedef struct benchmark_configuration_ts {
  int num_trials;
  int num_runs_per_trial;

  FILE *result_file;

  int min_size;
  int max_size;
  int step_size;

  int in_m0;
  int in_k0;
  int in_n0;
} benchmark_configuration_t;

/* Benchmark results */
typedef struct benchmark_results_ts {
  int num_trials;
  long *results;
} benchmark_results_t;

/* Function pointer typedefs for compute and model functions */
typedef void (benched_function_t)(
  op_params_t*,
  op_inputs_t*,
  op_outputs_t*,
  op_inouts_t*,
  hwctx_t*
);

typedef void (model_function_t)(
  op_model_t*,
  op_params_t*,
  op_inputs_t*,
  op_outputs_t*,
  op_inouts_t*,
  hwctx_t*
);

/* Standard function names for compute and model (test & reference) */
void COMPUTE_NAME_REF(
  op_params_t  *op_params,
  op_inputs_t  *inputs,
  op_outputs_t *outputs,
  op_inouts_t  *inouts,
  hwctx_t      *hwctx
);

void COMPUTE_NAME_TST(
  op_params_t  *op_params,
  op_inputs_t  *inputs,
  op_outputs_t *outputs,
  op_inouts_t  *inouts,
  hwctx_t      *hwctx
);

void COMPUTE_MODEL_NAME_REF(
  op_model_t   *model,
  op_params_t  *op_params,
  op_inputs_t  *inputs,
  op_outputs_t *outputs,
  op_inouts_t  *inouts,
  hwctx_t      *hwctx
);

void COMPUTE_MODEL_NAME_TST(
  op_model_t   *model,
  op_params_t  *op_params,
  op_inputs_t  *inputs,
  op_outputs_t *outputs,
  op_inouts_t  *inouts,
  hwctx_t      *hwctx
);

#endif /* _COMPUTE_H */
