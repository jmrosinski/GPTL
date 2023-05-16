/*
** $Id: private.h,v 1.74 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains definitions private to GPTL and inaccessible to invoking user environment
*/
#ifndef GPTL_DEVICE_H
#define GPTL_DEVICE_H

#include "config.h"
#include "devicehost.h"
#include <stdio.h>
// Need cuda.h for cudaGetErrorString() below
#include <cuda.h>

#define SUCCESS 0
#define FAILURE -1
#define NOT_ROOT_OF_WARP -2
#define WARPID_GT_MAXWARPS -3

// Flattening a 2d index into a 1d index gives good speedup 
#define FLATTEN_TIMERS(SUB1,SUB2) (SUB1)*maxtimers + (SUB2)

typedef struct {
  long long last;              // timestamp from last call
  long long accum;             // accumulated time
  long long max;               // longest time for start/stop pair
  long long min;               // shortest time for start/stop pair
} Wallstats;

typedef struct TIMER {
  Wallstats wall;              // wallclock stats
  unsigned long count;         // number of start/stop calls
#ifdef ENABLE_GPURECURSION
  uint recurselvl;             // recursion level
#endif
  uint smid;                   // SM the region is running on
  uint badsmid_count;          // number of times SM id changed
  uint negdelta_count;         // number of times a negative time increment occurred
  bool onflg;                  // timer currently on or off
} Timer;

typedef struct {
  char name[MAX_CHARS+1];
} Timername;

typedef struct {
  long long accum_max;
  long long accum_min;
  long long max;
  long long min;
  unsigned long count;
  unsigned int negdelta_count_max;
  int negdelta_count_max_warp;
  unsigned int negdelta_nwarps;
  int accum_max_warp;
  int accum_min_warp;
  int nwarps;
  int count_max;
  int count_max_warp;
  int count_min;
  int count_min_warp;
  uint badsmid_count;
  char name[MAX_CHARS+1];
} Gpustats;

namespace gpu {
  extern __device__ {
    // Variables used in multiple routines
    Timer *timers;            // array (also linked list) of timers
    Timername *timernames;    // array of timer names
    int max_name_len;         // max length of timer name
    int ntimers;              // number of timers
    int maxwarpid_found;      // number of warps found : init to 0 
    bool verbose;             // output verbosity                  
    double gpu_hz;            // clock freq                        
    int warps_per_sm;         // used for overhead calcs
#ifdef ENABLE_CONSTANTMEM
    __constant__ bool initialized; // GPTLinitialize has been called
    __constant__ int maxtimers;    // max number of timers allowed
    __constant__ int warpsize;     // warp size
    __constant__ int maxwarps;     // max number of warps that will be examined
#else
                 bool initialized; // GPTLinitialize has been called
                 int maxtimers;    // max number of timers allowed
                 int warpsize;     // warp size
                 int maxwarps;     // max number of warps that will be examined
#endif
    // Function prototypes used in multiple routines
    inline int get_warp_num (void);         // get 0-based 1d warp number
    inline void update_stats_gpu (const int, Timer *, const long long, const int, const uint);
  }
}

// For error checking on GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess) 
      {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
      }
  }
#endif
