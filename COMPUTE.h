#ifndef _COMPUTE_H
#define _COMPUTE_H

/*
  NOTE:OPERATION_SPECIFIC
  Checklist for adding a new operation:
  0. Structures

  1. Methods (which structs do they depend on)


  HARDWARE_TARGET_SPECIFIC
  checklist for calling an external function (messing with the build)
  
  Checklist for adding new hardware:
 */



// operation specific parameters
typedef struct op_params_ts {
  int m;
  int n;
  int k;

  int rs_a;
  int cs_a;

  int rs_b;
  int cs_b;

  int rs_c;
  int cs_c;

  int m0;
  int n0;
} op_params_t;

typedef struct op_inputs_ts {
  float *A;
  float *B;
} op_inputs_t;

typedef struct op_outputs_ts {
  float *C;
} op_outputs_t;

typedef struct op_inouts_ts {
  float *inout;
} op_inouts_t;

typedef struct op_model_ts {
  float flops;
  float bytes;
} op_model_t;


/*
  TODO: Should be in it's own header.
  
  Hardware context

  Could inlcude:
  + Distributed memory node
  (Physical Topology, virtual topology via Comm Groups)
  + Shared memory node
  (hostname)
  (#threads, numa domains, kaffinity, low-level memory organization: ctrls, channels, ranks, banks, rows, cols)
  + Core node (Kernels x Memory Movement x Comm)
  (HW threads, Vector Units, #ports, registers)
  + Acclerator
  (maybe just it's own shared node on a dist memory system....)
  +
  
*/
typedef struct hwctx_ts {
  int total_available_threads;
} hwctx_t;

/*
  TODO: Operation Implementation specific, should be moved.

  // execution plan
  typedef struct op_implementation_ts { } op_implementation_t;

*/


/*
  TODO: Should be somewhere else.
  
*/

typedef struct benchmark_configuration_ts
{
    // Experimental parameters
    int num_trials;
    int num_runs_per_trial;

    // What we will output to
    FILE *result_file;

    // General Problem parameters
    int min_size;
    int max_size;
    int step_size;

    // NOTE:OPERATION_SPECIFIC
    int in_m0; // rows of A and C
    int in_k0; // shared inner dimension (columns of A, rows of B)
    int in_n0; // columns of B and C

} benchmark_configuration_t;


typedef struct benchmark_results_ts
{
  int num_trials;
  long *results;
} benchmark_results_t;


/*
void (*benched_function_tt)( op_params_t  *op_params,
			     op_inputs_t  *inputs,
			     op_outputs_t *outputs,
			     op_inouts_t  *inouts,
			     hwctx_t   *hwctx );

typedef benched_function_tt benched_function_t;


typedef double (model_function_t)( op_params_t*,
				   op_inputs_t*,
				   op_outputs_t*,
				   op_inouts_t*,
				   hwctx_t*);
*/


typedef void (benched_function_t)( op_params_t*,
				    op_inputs_t*,
				    op_outputs_t*,
				    op_inouts_t*,
				    hwctx_t*);

typedef void (model_function_t)( op_model_t*,
				 op_params_t*,
				 op_inputs_t*,
				 op_outputs_t*,
				 op_inouts_t*,
				 hwctx_t*);



void COMPUTE_NAME_REF( op_params_t  *op_params,
		       op_inputs_t  *inputs,
		       op_outputs_t *outputs,
		       op_inouts_t  *inouts,
		       hwctx_t   *hwctx );

void COMPUTE_NAME_TST( op_params_t  *op_params,
		       op_inputs_t  *inputs,
		       op_outputs_t *outputs,
		       op_inouts_t  *inouts,
		       hwctx_t   *hwctx );


void COMPUTE_MODEL_NAME_TST( op_model_t   *model,
			     op_params_t  *op_params,
			     op_inputs_t  *inputs,
			     op_outputs_t *outputs,
			     op_inouts_t  *inouts,
			     hwctx_t   *hwctx );

void COMPUTE_MODEL_NAME_REF( op_model_t   *model,
			     op_params_t  *op_params,
			     op_inputs_t  *inputs,
			     op_outputs_t *outputs,
			     op_inouts_t  *inouts,
			     hwctx_t   *hwctx );




#endif /* _COMPUTE_H */
