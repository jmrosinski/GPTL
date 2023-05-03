#include "config.h"  // Must be first include.
#include "device.h"
#include <stdio.h>
#include <cuda.h>

extern "C" {

// Return useful GPU properties. Use arg list for SMcount, cores_per_sm, and cores_per_gpu even 
// though they're globals, because this is a user-callable routine
__host__ int GPTLget_gpu_props (int *khz, int *warpsize, int *devnum, int *SMcount,
				int *cores_per_sm, int *cores_per_gpu)
{
  cudaDeviceProp prop;
  cudaError_t err;
  static const char *thisfunc = "GPTLget_gpu_props";

  if ((err = cudaGetDeviceProperties (&prop, 0)) != cudaSuccess) {
    printf ("%s: error:%s", thisfunc, cudaGetErrorString (err));
    return -1;
  }

  *khz      = prop.clockRate;
  *warpsize = prop.warpSize;
  *SMcount  = prop.multiProcessorCount;

  // Begin code derived from stackoverflow to determine cores_per_sm
  // If helper_cuda.h and cuda_runtime.h is available (it's not currently in PATH)
  // could call _ConvertSMVer2Cores(prop.major, prop.minor) to get cores_per_sm
  // probably also need cuda_runtime.h
  switch (prop.major){
  case 2: // Fermi
    if (prop.minor == 1)
      *cores_per_sm = 48;
    else
      *cores_per_sm = 32;
    break;
  case 3: // Kepler
    *cores_per_sm = 192;
    break;
  case 5: // Maxwell
    *cores_per_sm = 128;
    break;
  case 6: // Pascal
    if ((prop.minor == 1) || (prop.minor == 2))
      *cores_per_sm = 128;
    else if (prop.minor == 0)
      *cores_per_sm = 64;
    else
      printf("Unknown device type\n");
    break;
  case 7: // Volta and Turing
    if ((prop.minor == 0) || (prop.minor == 5))
      *cores_per_sm = 64;
    else
      printf("Unknown device type\n");
    break;
  case 8: // Ampere
    if (prop.minor == 0)
      *cores_per_sm = 64;
    else
      printf("Unknown device type\n");
    break;
  default:
    printf("Unknown device type\n"); 
    break;
  }
  // End code derived from stackoverflow to determine cores_per_sm
  
  // Use _ConvertSMVer2Cores when it is available from nvidia
  //  cores_per_gpu = _ConvertSMVer2Cores (prop.major, prop.minor) * prop.multiProcessorCount);
  *cores_per_gpu = *cores_per_sm * (*SMcount);
  
  err = cudaGetDevice (devnum);  // device number
  return 0;
}

__host__ int GPTLcudadevsync (void)
{
  cudaDeviceSynchronize ();
  return 0;
}

// The need for these 2 wrapping functions enables gptl.c above to be built with a pure C compiler
// and therefore not require a .cu extension, which itself can cause problems when CUDA is not
// in play.
__host__ int GPTLreset_all_gpu_fromhost (void)
{
  static int *global_retval = 0; // return code from __global__ function

  if (global_retval == 0)  // Unallocated means first call
    // Create space for a "return" value for __global__functions to be checked on CPU
    gpuErrchk (cudaMallocManaged (&global_retval, sizeof (int)));

  *global_retval = 0;
  GPTLreset_all_gpu <<<1,1>>> (global_retval);
  cudaDeviceSynchronize ();
  if (*global_retval != 0)
    printf ("GPTLreset_all_gpu_fromhost: Failure from GPTLreset_all_gpu\n");
  return *global_retval;
}

__host__ int GPTLfinalize_gpu_fromhost (void)
{
  GPTLfinalize_gpu <<<1,1>>> ();
  return 0;
}
}
