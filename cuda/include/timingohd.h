#ifndef GPTLTIMINGOHD_H
#define GPTLTIMINGOHD_H

#ifdef TIME_GPTL
#define NUM_INTERNAL_TIMERS 3
namespace timingohd {
  extern __device__ const char *internal_name[NUM_INTERNAL_TIMERS];
  extern __device__ const long long *globcount;            // for timing GPTL itself
  // Indices for internal timers
  extern __device__ const int istart;
  extern __device__ const int istop;
  extern __device__ const int update_stats;
}
#endif
#endif
