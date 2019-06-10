#include <cuda.h>
#include "../cuda/gptl_cuda.h"
__global__ void sleep (float seconds, int outerlooplen)
{
  int ret;
  int blockId;
  int n;

  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  if (n < outerlooplen) {
    ret = GPTLmy_sleep (seconds);
  }
}
