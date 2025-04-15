#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "COMPUTE.h"

#include "timer.h"
#include "utils.h"




void time_function_under_test( benched_function_t function_under_test,
			       benchmark_configuration_t *bench_config,
			       benchmark_results_t *bench_results,
			       op_params_t  *op_params,
			       op_inputs_t  *inputs,
			       op_outputs_t *outputs,
			       op_inouts_t  *inouts,
			       hwctx_t   *hwctx )
{
  // Initialize the start and stop variables.
  TIMER_INIT_COUNTERS(stop, start);

  // Click the timer a few times so the subsequent measurements are more accurate
  TIMER_WARMUP(stop,start);

  // flush the cache
  flush_cache();
  
  for(int trial = 0; trial < bench_config->num_trials; ++trial )
    {

      /*
	Time code.
      */
      // start timer
      TIMER_GET_CLOCK(start);

      ////////////////////////
      // Benchmark the code //
      ////////////////////////

      for(int runs = 0; runs < bench_config->num_runs_per_trial; ++runs )
	{
	  function_under_test( op_params,
			       inputs,
			       outputs,
			       inouts,
			       hwctx );
	}

      ////////////////////////
      // End Benchmark      //
      ////////////////////////

        
      // stop timer
      TIMER_GET_CLOCK(stop);

      // subtract the start time from the stop time
      TIMER_GET_DIFF(start,stop,bench_results->results[trial]);
    }
}


void op_params_init(int p, benchmark_configuration_t *bench_config, op_params_t *op_params)
{
    op_params->m0 = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_m0);
    op_params->k  = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_k0); // inner dimension
    op_params->n0 = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_n0);

    op_params->m = op_params->m0;
    op_params->n = op_params->n0;

    // Row-major layout
    op_params->rs_a = op_params->k;
    op_params->cs_a = 1;

    op_params->rs_b = op_params->n;
    op_params->cs_b = 1;

    op_params->rs_c = op_params->n;
    op_params->cs_c = 1;
}


void op_inputs_init(op_params_t *op_params, op_inputs_t *op_inputs)
{
    int A_sz = op_params->m * op_params->k;
    int B_sz = op_params->k * op_params->n;

    op_inputs->A = (float *)malloc(sizeof(float) * A_sz);
    op_inputs->B = (float *)malloc(sizeof(float) * B_sz);

    fill_buffer_with_random(A_sz, op_inputs->A);
    fill_buffer_with_random(B_sz, op_inputs->B);
}


void op_outputs_init(op_params_t *op_params, op_outputs_t *op_outputs)
{
    int C_sz = op_params->m * op_params->n;
    op_outputs->C = (float *)malloc(sizeof(float) * C_sz);
    fill_buffer_with_value(C_sz, 0.0f, op_outputs->C);
}


void op_inouts_init(op_params_t *op_params, op_inouts_t *op_inouts)
{
}


void init_benchmark_results(benchmark_configuration_t *bench_config, benchmark_results_t *bench_results)
{
      
  bench_results->results = (long *)malloc(sizeof(long)*bench_config->num_trials);
  bench_results->num_trials = bench_config->num_trials;
}


// TODO: HERE
void compute_benchmark_results( benchmark_configuration_t *bench_config,
				benchmark_results_t *bench_results,
				model_function_t *fun_compute_model,
				op_params_t  *op_params,
				op_inputs_t  *op_inputs_tst,
				op_outputs_t *op_outputs_tst,
				op_inouts_t  *op_inouts_tst,
				hwctx_t      *hw_ctx )
{
      long min_res = pick_min_in_list(bench_results->num_trials, bench_results->results);
      float nanoseconds = ((float)min_res)/(bench_config->num_runs_per_trial);

      op_model_t model;

      fun_compute_model( &model,
			op_params,
			op_inputs_tst,
			op_outputs_tst,
			op_inouts_tst,
			hw_ctx );
      
      
      // This gives us throughput as GFLOP/s
      double flop   =  model.flops;
      
      double throughput   =  flop / nanoseconds;
      double bytes        =  model.bytes;

      double gbytes_per_s =  bytes/ nanoseconds;

      // fprintf(bench_config.result_file,
      // "size,m,n,flop,throughput,bytes,GB_per_s,nanoseconds\n");
      fprintf(bench_config->result_file,
	      "%i,%i,%i,"
	      "%2.3e,%2.3e,"
	      "%2.3e,%2.3e,"
	      "%2.3e\n",
      	      op_params->m0,
	      op_params->m0,
	      op_params->n0,
	      flop, throughput,
	      bytes, gbytes_per_s,
	      nanoseconds);
  
}

void bench_config_init( int argc, char *argv[], benchmark_configuration_t *bench_config)
{
  // TODO: Parameterize these
  bench_config->num_trials = 10;
  bench_config->num_runs_per_trial = 10;
  
  // Get command line arguments
  if(argc == 1 )
    {
      bench_config->min_size  = 16;
      bench_config->max_size  = 1524;
      bench_config->step_size = 16;

      // defaults
      bench_config->in_m0=1;
      bench_config->in_n0=1;
      bench_config->in_k0=1;

      // default to printing to stdout
      bench_config->result_file = stdout;
    }
    else if(argc == 6 + 1 || argc == 7 + 1)
    {
        bench_config->min_size  = atoi(argv[1]);
        bench_config->max_size  = atoi(argv[2]);
        bench_config->step_size = atoi(argv[3]);
    
        bench_config->in_m0 = atoi(argv[4]);
        bench_config->in_k0 = atoi(argv[5]);
        bench_config->in_n0 = atoi(argv[6]);
    
        bench_config->result_file = stdout;
    
        if(argc == 7 + 1)
            bench_config->result_file = fopen(argv[7], "w");
    }
    else
    {
        printf("usage: %s min max step m0 k0 n0 [filename]\n", argv[0]);
        exit(1);
    }
}

int main( int argc, char *argv[] )
{
  
  /* TODO: MOVE
     command line argv --> hw_ctx (plen, vlen, funits, registers)
   */
  hwctx_t hw_ctx;
  /* END OF TODO*/

  
  // Problem parameters
  benchmark_configuration_t bench_config;
  bench_config_init( argc, argv, &bench_config);


  // print the first line of the output
  fprintf(bench_config.result_file, "size,m,n,flop,throughput,bytes,GB_per_s,nanoseconds\n");

  
  // step through all of the problem sizes of interest
  for( int p = bench_config.min_size;
       p < bench_config.max_size;
       p += bench_config.step_size )
    {

      op_params_t  op_params;
      op_inputs_t  op_inputs_tst;
      op_outputs_t op_outputs_tst;
      op_inouts_t  op_inouts_tst;
      
      // create a set of parameters for the current experiment.
      op_params_init(p,&bench_config, &op_params );
      
      
      // Initialize the data
      op_inputs_init(&op_params, &op_inputs_tst);
      op_outputs_init(&op_params, &op_outputs_tst);
      op_inouts_init(&op_params, &op_inouts_tst);

      
      // Perform the computation
      benchmark_results_t bench_results;
      init_benchmark_results(&bench_config,&bench_results);
      
     
      time_function_under_test( COMPUTE_NAME_TST,
				&bench_config,
				&bench_results,
				&op_params,
				&op_inputs_tst,
				&op_outputs_tst,
				&op_inouts_tst,
				&hw_ctx );


      /////////////////////
      // Compute the results

      model_function_t *model_fun = COMPUTE_MODEL_NAME_TST;
      compute_benchmark_results(&bench_config, &bench_results,
				model_fun,
				&op_params,
				&op_inputs_tst,
				&op_outputs_tst,
				&op_inouts_tst,
				&hw_ctx);
      
      ///////
      free(bench_results.results);
      ///////

      //////////////////
      // Free the  buffers
      free(op_inputs_tst.A);
      free(op_inputs_tst.B);
      free(op_outputs_tst.C);

    }


  // close the result file
  fclose(bench_config.result_file);


}
