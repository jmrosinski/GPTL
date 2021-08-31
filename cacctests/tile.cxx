#include "../config.h"
#include <openacc.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

// Test code for tiling
// Best to compile with C++ compiler due to multi-dim arrays--gdb can't handle under C
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

void printdiff (struct timeval, struct timeval, const char *);

  int main ()
{
  const int M = 1000;
  const int N = 1000;
  float A[N][M];
  float Anew[N][M];
  float maxerr;
  int isave, jsave;
  int ret;
  struct timeval tp1, tp2;

  for( int j=0; j < N; ++j)
    for(int i=0; i < M; ++i)
      A[j][i] = i + j;

  ret = gettimeofday (&tp1, 0);
#pragma acc enter data create (Anew) copyin (A)
  ret = gettimeofday (&tp2, 0);
  printdiff (tp1, tp2, "enter_data");

  ret = gettimeofday (&tp1, 0);
  // Tiling with g++ gives wrong answers nvc++ works ok
//#pragma acc parallel loop tile(32,4)
#pragma acc parallel loop gang 
  for (int j=1; j < N-1; ++j) {
#pragma acc loop worker vector
    for (int i=1; i < M-1; ++i) {
      Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
    }
  }
  ret = gettimeofday (&tp2, 0);
  printdiff (tp1, tp2, "parallel_loop");

  ret = gettimeofday (&tp1, 0);
#pragma acc exit data copyout (Anew)
  ret = gettimeofday (&tp2, 0);
  printdiff (tp1, tp2, "exit_data");

  maxerr = -1.;
  for (int j=1; j < N-1; ++j) {
    for (int i=1; i < M-1; ++i) {
      float anew = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
      float diff = fabs (Anew[j][i] - anew);
      if (diff > maxerr) {
	isave = i;
	jsave = j;
	maxerr = diff;
      }
    }
  }
  printf ("maxerr=%f at isave=%d jsave=%d A=%f %f %f %f\n", maxerr, isave, jsave,
	  A[jsave][isave+1], A[jsave][isave-1], A[jsave-1][isave], A[jsave+1][isave]);
  return 0;
}

void printdiff (struct timeval tp1, struct timeval tp2, const char *str)
{
  float diff = 1.e3*(tp2.tv_sec - tp1.tv_sec) + 1.e-3*(tp2.tv_usec - tp1.tv_usec);
  printf ("%f msec %s\n", diff, str);
}
