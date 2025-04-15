#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "COMPUTE.h"
#include "utils.h"

#if 0
// generic params
// op specific params
// impl specific input
// impl specific output
// impl specific inout
// ctx

// 0 get argument
// -- hw related #vec, # threads
// -- op generic: <log file> # runs # trials
// -- op specific
// Loop
// - 0. (op specific) Create buffer
// - 1. (op specific) Fill buffers


#endif






int main( int argc, char *argv[] )
{
  // What we will output to
  FILE *result_file;
  
  // Problem parameters
  int min_size;
  int max_size;
  int step_size;

  int in_m0;
  int in_n0;
  int in_k0;

  // Get command line arguments
  if(argc == 1 )
    {
      min_size  = 16;
      max_size  = 256;
      step_size = 16;

      // defaults
      in_m0=1;
      in_n0=1;
      in_k0=1;

      // default to printing to stdout
      result_file = stdout;
    }
  else if(argc == 6 + 1 || argc == 7 + 1 )
    {
      min_size  = atoi(argv[1]);
      max_size  = atoi(argv[2]);
      step_size = atoi(argv[3]);

      in_m0=atoi(argv[4]);
      in_n0=atoi(argv[5]);
      in_k0=atoi(argv[6]);

      // default to printing to stdout
      result_file = stdout;

      // If we were given a file, use that.
      if(argc == 7 + 1)
	      result_file = fopen(argv[7],"w");

    }
  else
    {
      //      argv    0   1   2    3  4  5  6       7
      printf("usage: %s min max step m0 n0 k0 [filename]\n",
	     argv[0]);
      exit(1);
    }


  // step through all of the problem sizes of interest
  for( int p = min_size;
       p < max_size;
       p += step_size )
    {

      // input sizes
      int m0=scale_p_on_pos_ret_v_on_neg(p,in_m0);
      int n0=scale_p_on_pos_ret_v_on_neg(p,in_n0);
      int k0=scale_p_on_pos_ret_v_on_neg(p,in_k0);

      int A_sz = m0 * k0;
      int B_sz = k0 * n0;
      int C_sz = m0 * n0;

      float *A_ref = malloc(sizeof(float) * A_sz);
      float *B_ref = malloc(sizeof(float) * B_sz);
      float *C_ref = malloc(sizeof(float) * C_sz);

      float *A_tst = malloc(sizeof(float) * A_sz);
      float *B_tst = malloc(sizeof(float) * B_sz);
      float *C_tst = malloc(sizeof(float) * C_sz);

      float *C_diffs = malloc(sizeof(float) * C_sz);

      // fill src_ref with random values
      fill_buffer_with_random(A_sz, A_ref);
      fill_buffer_with_random(B_sz, B_ref);
      fill_buffer_with_value(C_sz, 0.0f, C_ref);

     
      memcpy(A_tst, A_ref, sizeof(float) * A_sz);
      memcpy(B_tst, B_ref, sizeof(float) * B_sz);
      memcpy(C_tst, C_ref, sizeof(float) * C_sz);

      /*
	Run the reference
      */
      hwctx_t hw_ctx;
	    
      op_params_t op_params;
      op_params.m = m0;
      op_params.n = n0;
      op_params.k = k0;
      
      op_params.rs_a = k0;
      op_params.cs_a = 1;

      op_params.rs_b = n0;
      op_params.cs_b = 1;

      op_params.rs_c = n0;
      op_params.cs_c = 1;

      op_params.m0 = m0;
      op_params.n0 = n0;

      op_inputs_t op_inputs_ref = {
        .A = A_ref,
        .B = B_ref
      };
    
      op_outputs_t op_outputs_ref = {
        .C = C_ref
      };

      op_inouts_t op_inouts_ref;
      
      // Perform the computation
      COMPUTE_NAME_REF( &op_params,
			&op_inputs_ref,
			&op_outputs_ref,
			&op_inouts_ref,
			&hw_ctx );



      op_inputs_t op_inputs_tst = {
        .A = A_tst,
        .B = B_tst
      };

      op_outputs_t op_outputs_tst = {
        .C = C_tst
      };

      op_inouts_t op_inouts_tst;

      
      // run the test
      // Perform the computation
      COMPUTE_NAME_TST( &op_params,
			&op_inputs_tst,
			&op_outputs_tst,
			&op_inouts_tst,
			&hw_ctx );



      // Verify results
      float res = compute_pair_wise_diff(C_sz, 1, 1, 1, C_ref, C_tst, C_diffs);
      long counts = count_num_errors(C_sz, 1, 1, 1, C_diffs);

      fprintf(result_file, "%i, %li, %f, ", m0, counts, res);

      if (res > ERROR_THRESHOLD)
          fprintf(result_file, "FAIL\n");
      else
          fprintf(result_file, "PASS\n");

      free(A_ref);
      free(B_ref);
      free(C_ref);

      free(A_tst);
      free(B_tst);
      free(C_tst);
      free(C_diffs);

    }


  // close the result file
  fclose(result_file);


}
