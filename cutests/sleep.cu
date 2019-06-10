#include <cuda.h>
#include "../cuda/gptl_cuda.h"
__global__ void sleep (float seconds, int outerlooplen)
{
  int ret;
  int blockId;
  int n;
  int total_gputime, total_gputime2, sleep1;

  ret = GPTLinit_handle_gpu ("total_gputime",  &total_gputime);
  ret = GPTLinit_handle_gpu ("total_gputime2", &total_gputime2);
  ret = GPTLinit_handle_gpu ("sleep1",         &sleep1);

  ret = GPTLstart_gpu (total_gputime);
  ret = GPTLstart_gpu (total_gputime2);
  blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 

  n = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;

  if (n < outerlooplen) {
    ret = GPTLstart_gpu (sleep1);
    ret = GPTLmy_sleep (seconds);
    ret = GPTLstop_gpu (sleep1);
  }
  ret = GPTLstop_gpu (total_gputime2);
  ret = GPTLstop_gpu (total_gputime);
}
