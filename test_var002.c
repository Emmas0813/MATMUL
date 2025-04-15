/*
  This is a simple implementation of an unoptimized sort.

  - richard.m.veras@ou.edu

*/


#include <stdio.h>
#include <stdlib.h>

#include "instruments.h"

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef COMPUTE_FLOP_NAME
#define COMPUTE_FLOP_NAME baseline_flop
#endif

#ifndef COMPUTE_BYTES_NAME
#define COMPUTE_BYTES_NAME baseline_bytes
#endif

// Dimensions of the Filter
static const int R = 4;
static const int Q = 3;

/* Weights of the Filter.
 - We are going to linearize this 2d array in row major order.
 - The following code illustrates the layout of the data
   and the relationship between a 2D array and its linearized
   counterpart.

  int NUMROWS=2;
  int NUMROWS=3;
 
  Arr2D[NUMROWS][NUMCOLS]=
                       // j= 0  1  2
                           {{a, b, c}, //  <-- i = 0
                            {d, e, f}}; // <-- i = 1
  Arr1D[NUMROWS*NUMCOLS];

  for( i=0; i<NUMROWS; ++i)
    for( j=0; j<NUMCOLS; ++j)
      Arr1D[i*NUMCOLS+j] = Arr2D[i][j]

 
      //   Arr1D[NUMROWS*NUMCOLS] |--->
      //         j=0  1  2
      //          {a, b, d,  <-- i=0
      //	   e, f, g   <-- i=1

*/

//weights[(Q)][(R)]
//weights[(Q)*(R)]
static float weights[] =
  //r=0    1     2     3
  {-1.1, -1.1,  1.2, -2.1,  // q=0
   -1.1, -2.1, -1.2,  2.2,  // q=1
   -2.1,  0.1,  0.2,  1.2}; // q=2




double COMPUTE_FLOP_NAME( int m0, int n0 )
{
  return 2*m0*n0*(Q)*(R);
}

/*
  1 read           for x vector   (m*n)
  1 read + 1 write for y vector 2*(m*n)
  1 read for the weights (Q*R)
*/
double COMPUTE_BYTES_NAME( int m0, int n0 )
{
  return (3*(m0*n0)+ ((Q)*(R)))*sizeof(float);
}


/*
  X and Y are linearized as 1D arrays in memory
  with m0 number of rows and n0 number of columns.
  They are row major order, which means that elements
  on the same row, but adjacent columns are next to
  each other in memory.

  m0 == 4; 0 <= i < m0
  n0 == 7; 0 <= j < n0
  
  y[m0*n0]
  x[m0*n0]
  

 Logical view of X or Y 
   \ j=0 1 2 3 4 5 6 n0
    \ _______________
i= 0 | a b c d e f g
 = 1 | h i j k l m n
 = 2 | o p q r s t u
 = 3 | v q x y z A B
   m0

Indices to Addresses
   
 i | j | &x[i*n0+j]
 __________________
 0 | 0 | &x[0]
 0 | 1 | &x[1]
 0 | 6 | &x[6]
 1 | 0 | &x[7]
 2 | 0 | &x[14]
 2 | 1 | &x[15]
 3 | 6 | &x[27]

 

  Layout in memory
 
 Addr   | Value
 ______________
 &x[0]  | a
 &x[1]  | b
 &x[3]  | c
 &x[4]  | d
 &x[5]  | e
 &x[6]  | f
 &x[7]  | g
 &x[8]  | h
 &x[14] | o
 &x[15] | p
 &x[27] | B
  
 */




void COMPUTE_NAME( int m0,
		   int n0,
		   float *x,
		   float *y )

{
  BEGIN_INSTRUMENTATION;
  for( int i0 = 0; i0 < m0; ++i0 )
    {
      y[i0]=y[i0]*2+1;
    }
  END_INSTRUMENTATION;
}


