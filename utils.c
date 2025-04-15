/*

  Helper and auxilliary routines for the verifier and timer.

  richar.m.veras@ou.edu

 */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"


void fill_buffer_with_random( int num_elems, float *buff )
{
  //long long range = RAND_MAX;  
  int upper = HALF_RANGE;
  int lower = -(HALF_RANGE);
  int range = upper - lower;
  
  for(int i = 0; i < num_elems; ++i)
    {
      //buff[i] = ((float)(rand()-((range)/2)))/((float)range);
      int rval = rand();
      int bounded_val = rval%(range+1)+lower;
      float scaled  = ((float)bounded_val)/((float)range);
      buff[i] = scaled;
    }
}

void fill_buffer_with_value( int num_elems, float val, float *buff )
{
  for(int i = 0; i < num_elems; ++i)
    buff[i] = val;
}


long count_num_errors(int m, int n, int rs, int cs, float *a)
{
  long counts = 0;

  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      {

	if(a[i*rs+j*cs]>ERROR_THRESHOLD)
	  counts++;
      }

  return counts;

}

float compute_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b, float *c)
{
  float max_diff = 0.0;

  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      {
	
	float sum  = fabs(a[i*rs+j*cs]+b[i*rs+j*cs]);
	float diff = fabs(a[i*rs+j*cs]-b[i*rs+j*cs]);

	float res = 0.0f;

	if(fabs(sum) < ERROR_THRESHOLD)
	  res = fabs(diff);
	else
	  res = 2*diff/sum;

	/*
	if(res > ERROR_THRESHOLD)
	  printf("%i, %f %f %f %f %f \n",i, a[i*rs+j*cs],b[i*rs+j*cs],sum,diff,res);
	*/

	c[i*rs+j*cs] = res;
	
	if( res > max_diff )
	  max_diff = res;

      }

  return max_diff;
}


float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b)
{
  float max_diff = 0.0;

  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      {
	float sum  = fabs(a[i*rs+j*cs]+b[i*rs+j*cs]);
	float diff = fabs(a[i*rs+j*cs]-b[i*rs+j*cs]);

	float res = 0.0f;

	if(sum == 0.0f)
	  res = diff;
	else
	  res = 2*diff/sum;

	if( res > max_diff )
	  max_diff = res;
      }

  return max_diff;
}

void print_vector( int m, float *x )
{
  for(int i = 0; i <m; ++i)
    printf("%f\n",x[i]);
}


int scale_p_on_pos_ret_v_on_neg(int p, int v)
{
  if (v < 1)
    return -1*v;
  else
    return v*p;
}

long pick_min_in_list(int num_trials, long *results)
{
  long current_min = LONG_MAX;

  for( int i = 0; i < num_trials; ++i )
    if( results[i] < current_min )
      current_min = results[i];

  return current_min;
}

void flush_cache()
{
  
  int size = 1024*1024*8;

  int *buff = (int *)malloc(sizeof(int)*size);
  int i, result = 0;
  volatile int sink;
  for (i = 0; i < size; i ++)
    result += buff[i];
  sink = result; /* So the compiler doesn't optimize away the loop */

  free(buff);
}
