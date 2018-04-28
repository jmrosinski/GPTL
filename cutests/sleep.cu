#include <cuda.h>
#include "../cuda/gptl_cuda.h"
__global__ void sleep (float seconds, int outerlooplen)
{
  int ret;
  int blockId;
  int n;

  ret = GPTLstart_gpu ("total_gputime");
  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  if (n < outerlooplen) {
    ret = GPTLstart_gpu ("sleep1");
    ret = GPTLmy_sleep (seconds);
    ret = GPTLstop_gpu ("sleep1");
  }
  ret = GPTLstop_gpu ("total_gputime");
}
