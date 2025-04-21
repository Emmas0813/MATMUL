/*

  Helper and auxilliary routines for the verifier and timer.

  richar.m.veras@ou.edu

 */

// Error bounds
#define ERROR_THRESHOLD 1e-3
#define HALF_RANGE 1000
#define ABS_TOLERANCE 1e-6f


void fill_buffer_with_random( int num_elems, float *buff );
void fill_buffer_with_value( int num_elems, float val, float *buff );
long count_num_errors(int m, int n, int rs, int cs, float *a);
float compute_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b, float *c);
float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b);
void print_vector( int m, float *x );
int scale_p_on_pos_ret_v_on_neg(int p, int v);

long pick_min_in_list(int num_trials, long *results);
void flush_cache();
