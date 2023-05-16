#ifdef TIME_GPTL
#define NUM_INTERNAL_TIMERS 3
namespace timingohd {
  extern __device__ {
    const char *internal_name[NUM_INTERNAL_TIMERS];
    const long long *globcount;            // for timing GPTL itself
    // Indices for internal timers
    const int istart;
    const int istop;
    const int update_stats;
  }
}
#endif
