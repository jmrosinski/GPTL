#ifndef GPTLINITFINAL_H
#define GPTLINITFINAL_H
namespace init_final {
#ifdef ENABLE_CONSTANTMEM
  extern __device__    __constant__ bool initialized; // GPTLinitialize has been called
  extern __device__    __constant__ int maxtimers;    // max number of timers allowed
  extern __device__    __constant__ int warpsize;     // warp size
  extern __device__    __constant__ int maxwarps;     // max number of warps that will be examined
#else
  extern __device__                 bool initialized; // GPTLinitialize has been called
  extern __device__                 int maxtimers;    // max number of timers allowed
  extern __device__                 int warpsize;     // warp size
  extern __device__                 int maxwarps;     // max number of warps that will be examined
#endif
}
#endif
