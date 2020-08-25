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
  size_t size;
  cudaError_t err;
  static const size_t onemb = 1024 * 1024;
  //static const size_t heap_mb = 8;  // this number should avoid needing to reset the limit
  //static const size_t heap_mb = 128;
  static const char *thisfunc = "GPTLget_gpu_props";

  if ((err = cudaGetDeviceProperties (&prop, 0)) != cudaSuccess) {
    printf ("%s: error:%s", thisfunc, cudaGetErrorString (err));
    return -1;
  }

  *khz      = prop.clockRate;
  *warpsize = prop.warpSize;
  *SMcount  = prop.multiProcessorCount;

  // Begin code derived from stackoverflow to determine cores_per_sm
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
  
  printf ("%s: major.minor=%d.%d\n", thisfunc, prop.major, prop.minor);
  printf ("%s: SM count=%d\n",      thisfunc, *SMcount);
  printf ("%s: cores per sm=%d\n",  thisfunc, *cores_per_sm);
  printf ("%s: cores per GPU=%d\n", thisfunc, *cores_per_gpu);

  err = cudaGetDevice (devnum);  // device number
  err = cudaDeviceGetLimit (&size, cudaLimitMallocHeapSize);
  printf ("%s: default cudaLimitMallocHeapSize=%d MB\n", thisfunc, (int) (size / onemb));
  return 0;
}

__host__ int GPTLcompute_chunksize (const int oversub, const int inner_iter_count,
				    const int cores_per_gpu)
{
  int chunksize;
  float oversub_factor;
  static const char *thisfunc = "GPTLcompute_chunksize";

  if (oversub < 1)
    return GPTLerror ("%s: oversub=%d must be > 0\n", thisfunc, oversub);

  if (cores_per_gpu < 1)
    return GPTLerror ("%s: Input arg cores_per_gpu=%d means it hasn't been set or is bad\n",
		      thisfunc, cores_per_gpu);

  chunksize = (oversub * cores_per_gpu) / inner_iter_count;
  if (chunksize < 1) {
    chunksize = 1;
    oversub_factor = (float) inner_iter_count / (float) cores_per_gpu;
    printf ("%s: WARNING: chunksize=1 still results in an oversubscription factor=%f"
	    "compared to request=%d\n", thisfunc, oversub_factor, oversub);
  }
  return chunksize;
}

__host__ int GPTLcudadevsync (void)
{
  cudaDeviceSynchronize ();
  return 0;
}

// The need for these 2 wrapping functions enables gptl.c above to be built with a pure C compiler
// and therefore not require a .cu extension, which itself can cause problems when CUDA is not
// in play.
__host__ int GPTLreset_gpu_fromhost (void)
{
  GPTLreset_gpu <<<1,1>>> ();
  return 0;
}

__host__ int GPTLfinalize_gpu_fromhost (void)
{
  GPTLfinalize_gpu <<<1,1>>> ();
  return 0;
}
}
