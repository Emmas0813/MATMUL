#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "COMPUTE.h"
#include "timer.h"
#include "utils.h"

// Benchmark the function under test
void time_function_under_test(benched_function_t function_under_test,
                               benchmark_configuration_t *bench_config,
                               benchmark_results_t *bench_results,
                               op_params_t *op_params,
                               op_inputs_t *inputs,
                               op_outputs_t *outputs,
                               op_inouts_t *inouts,
                               hwctx_t *hwctx)
{
    TIMER_INIT_COUNTERS(stop, start);
    TIMER_WARMUP(stop, start);
    flush_cache();

    for (int trial = 0; trial < bench_config->num_trials; ++trial)
    {
        TIMER_GET_CLOCK(start);

        for (int run = 0; run < bench_config->num_runs_per_trial; ++run)
        {
            function_under_test(op_params, inputs, outputs, inouts, hwctx);
        }

        TIMER_GET_CLOCK(stop);
        TIMER_GET_DIFF(start, stop, bench_results->results[trial]);
    }
}

// Initialize matrix multiplication parameters
void op_params_init(int p, benchmark_configuration_t *bench_config, op_params_t *op_params)
{
    op_params->m0 = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_m0);
    op_params->k  = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_k0);
    op_params->n0 = scale_p_on_pos_ret_v_on_neg(p, bench_config->in_n0);

    op_params->m = op_params->m0;
    op_params->n = op_params->n0;

    op_params->rs_a = op_params->k;
    op_params->cs_a = 1;
    op_params->rs_b = op_params->n;
    op_params->cs_b = 1;
    op_params->rs_c = op_params->n;
    op_params->cs_c = 1;
}

// Allocate and initialize input matrices A and B
void op_inputs_init(op_params_t *op_params, op_inputs_t *op_inputs)
{
    int A_sz = op_params->m * op_params->k;
    int B_sz = op_params->k * op_params->n;

    op_inputs->A = malloc(sizeof(float) * A_sz);
    op_inputs->B = malloc(sizeof(float) * B_sz);

    fill_buffer_with_random(A_sz, op_inputs->A);
    fill_buffer_with_random(B_sz, op_inputs->B);
}

// Allocate and zero the output matrix C
void op_outputs_init(op_params_t *op_params, op_outputs_t *op_outputs)
{
    int C_sz = op_params->m * op_params->n;
    op_outputs->C = malloc(sizeof(float) * C_sz);
    fill_buffer_with_value(C_sz, 0.0f, op_outputs->C);
}

// Empty init for inouts (not used)
void op_inouts_init(op_params_t *op_params, op_inouts_t *op_inouts) {}

// Allocate result buffer for timing
void init_benchmark_results(benchmark_configuration_t *bench_config, benchmark_results_t *bench_results)
{
    bench_results->results = malloc(sizeof(long) * bench_config->num_trials);
    bench_results->num_trials = bench_config->num_trials;
}

// Compute performance metrics
void compute_benchmark_results(benchmark_configuration_t *bench_config,
                                benchmark_results_t *bench_results,
                                model_function_t *fun_compute_model,
                                op_params_t *op_params,
                                op_inputs_t *inputs,
                                op_outputs_t *outputs,
                                op_inouts_t *inouts,
                                hwctx_t *hwctx)
{
    long min_time = pick_min_in_list(bench_results->num_trials, bench_results->results);
    float nanoseconds = (float)min_time / bench_config->num_runs_per_trial;

    op_model_t model;
    fun_compute_model(&model, op_params, inputs, outputs, inouts, hwctx);

    double gflops = model.flops / nanoseconds;
    double gbytes = model.bytes / nanoseconds;

    fprintf(bench_config->result_file,
            "%d,%d,%d,%2.3e,%2.3e,%2.3e,%2.3e,%2.3e\n",
            op_params->m0, op_params->m, op_params->n,
            model.flops, gflops,
            model.bytes, gbytes,
            nanoseconds);
}

// Parse command-line arguments into benchmark config
void bench_config_init(int argc, char *argv[], benchmark_configuration_t *bench_config)
{
    bench_config->num_trials = 10;
    bench_config->num_runs_per_trial = 10;

    if (argc == 1)
    {
        bench_config->min_size = 16;
        bench_config->max_size = 512;
        bench_config->step_size = 16;
        bench_config->in_m0 = bench_config->in_k0 = bench_config->in_n0 = 1;
        bench_config->result_file = stdout;
    }
    else if (argc == 7 || argc == 8)
    {
        bench_config->min_size = atoi(argv[1]);
        bench_config->max_size = atoi(argv[2]);
        bench_config->step_size = atoi(argv[3]);

        bench_config->in_m0 = atoi(argv[4]);
        bench_config->in_k0 = atoi(argv[5]);
        bench_config->in_n0 = atoi(argv[6]);

        bench_config->result_file = (argc == 8) ? fopen(argv[7], "w") : stdout;

        if (argc == 8 && !bench_config->result_file)
        {
            perror("Failed to open output file");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Usage: %s min max step m0 k0 n0 [filename]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    hwctx_t hw_ctx;
    benchmark_configuration_t bench_config;
    bench_config_init(argc, argv, &bench_config);

    fprintf(bench_config.result_file, "size,m,n,flop,throughput,bytes,GB_per_s,nanoseconds\n");

    for (int p = bench_config.min_size; p < bench_config.max_size; p += bench_config.step_size)
    {
        op_params_t op_params;
        op_inputs_t inputs;
        op_outputs_t outputs;
        op_inouts_t inouts;

        op_params_init(p, &bench_config, &op_params);
        op_inputs_init(&op_params, &inputs);
        op_outputs_init(&op_params, &outputs);
        op_inouts_init(&op_params, &inouts);

        benchmark_results_t bench_results;
        init_benchmark_results(&bench_config, &bench_results);

        time_function_under_test(COMPUTE_NAME_TST,
                                 &bench_config,
                                 &bench_results,
                                 &op_params,
                                 &inputs,
                                 &outputs,
                                 &inouts,
                                 &hw_ctx);

        compute_benchmark_results(&bench_config, &bench_results,
                                  COMPUTE_MODEL_NAME_TST,
                                  &op_params,
                                  &inputs,
                                  &outputs,
                                  &inouts,
                                  &hw_ctx);

        free(bench_results.results);
        free(inputs.A);
        free(inputs.B);
        free(outputs.C);
    }

    fclose(bench_config.result_file);
    return 0;
}
