/*
** $Id: private.h,v 1.74 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** Contains definitions private to GPTL and inaccessible to invoking user environment
*/

#include "../devicehost.h"

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define STRMATCH(X,Y) (my_strcmp((X),(Y)) == 0)

#define NOT_ROOT_OF_WARP -2
#define WARPID_GT_MAXWARPS -3

typedef struct {
  long long last;              // timestamp from last call
  long long accum;             // accumulated time
  long long max;               // longest time for start/stop pair
  long long min;               // shortest time for start/stop pair
} Wallstats;

typedef struct TIMER {
  Wallstats wall;              // wallclock stats
  unsigned long count;         // number of start/stop calls
  uint negcount_stop;          // number of times a stop time < start time
  uint recurselvl;             // recursion level
  uint smid;                   // SM the region is running on
  uint badsmid_count;          // number of times SM id changed
  uint negdelta_count;         // number of times a negative time increment occurred
  bool onflg;                  // timer currently on or off
} Timer;

typedef struct {
  char name[MAX_CHARS+1];
} Timername;

// Function prototypes
extern "C" {
  /* These are callable from within gptl.cu */
__device__ extern int GPTLerror_1s (const char *, const char *);
__device__ extern int GPTLerror_2s (const char *, const char *, const char *);
__device__ extern int GPTLerror_3s (const char *, const char *, const char *, const char *);
__device__ extern int GPTLerror_1s1d (const char *, const char *, const int);
__device__ extern int GPTLerror_2s1d (const char *, const char *, const char *, const int);
__device__ extern int GPTLerror_2s2d (const char *, const char *, const char *, const int, const int);
__device__ extern int GPTLerror_1s2d (const char *, const char *, const int, const int);
__device__ extern int GPTLerror_1s1d1s (const char *, const char *, const int, const char *);
__device__ extern void GPTLreset_errors_gpu (void);                  /* num_errors to zero */
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess) 
      {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
      }
  }
