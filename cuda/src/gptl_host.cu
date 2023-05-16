#include "config.h"  // Must be first include.
#include "api.h"
#include <stdio.h>
#include <cuda.h>

// All routines in this file are either user-callable and/or called from gptl.c
// So C++ name-mangling must be disabled

extern "C" {

// GPTLget_gpu_props: User-callable routine returns useful GPU properties. 
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

// GPTLcudadevsync: User-callable routine to sync the GPU. 
__host__ int GPTLcudadevsync (void)
{
  cudaDeviceSynchronize ();
  return 0;
}
}
