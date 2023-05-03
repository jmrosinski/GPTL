/*
** gptl.cu
** Author: Jim Rosinski
**
** Main file contains most CUDA GPTL functions
*/

#include "config.h" // Must be first include.

#include <stdio.h>
#include <string.h>        // memcpy
#include <stdint.h>        // uint types
#include <cuda.h>
#include <limits.h>        // LLONG_MAX

#include "device.h"
#include "gptl_cuda.h"

extern "C" {

__device__ int GPTLinit_handle_gpu (const char *name,
				    int *handle)
{
  *handle = 1;
  return SUCCESS;
}

__device__ int GPTLstart_gpu (const int handle)
{
  return SUCCESS;
}

__device__ int GPTLstop_gpu (const int handle)
{
  return SUCCESS;
}

__device__ int GPTLget_wallclock_gpu (const int handle,
				      double *accum, double *max, double *min)
{
  *accum = 0.;
  *max = 0.;
  *min = 0.;
  return 0;
}

__device__ int GPTLmy_sleep (float seconds)
{
  return SUCCESS;
}

__device__ void GPTLdummy_gpu ()
{
  return;
}
}
