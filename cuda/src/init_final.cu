#include "config.h"     // Must be first include.
#include "api.h"
#include "util.h"
#include "timingohd.h"
#include <limits.h>     // LLONG_MAX

// File-local variables
static int *retval;  // Return value passed to CPU from __global__ routines

// Local function prototypes
__global__ static void initialize_gpu (const int, const int, const double, Timer *,
				       Timername *, const int, long long *, int *);
__global__ void finalize_gpu (int *);

// GPTLinitialize_gpu and GPTLfinalize_gpu are called from host so do not name mangle
extern "C" {
  /*
  ** GPTLinitialize_gpu: Initialize GPTL CUDA code. Calls initialize_gpu <<<1,1>>>
  */
  __host__ int GPTLinitialize_gpu (const int verbose_in,
				   const int maxwarps_in,
				   const int maxtimers_in,
				   const double gpu_hz_in,
				   const int warpsize_in,
				   const int cores_per_sm_in)
  {
    size_t nbytes;  // number of bytes to allocate
    // Issue cudaMalloc from CPU, and pass address to GPU to avoid mem problems: When run from
    // __global__ routine, mallocable memory is severely decreased for some reason.
    static Timer *timers_cpu = 0;          // array of timers
    static Timername *timernames_cpu = 0;  // array of timer names
    static long long *globcount_cpu = 0;   // for internally timing GPTL

    // Set constant memory values: First arg is pass by reference so no "&"
    nbytes = maxwarps_in * maxtimers_in * sizeof (Timer);
    gpuErrchk (cudaMalloc (&timers_cpu, nbytes));
#ifdef ENABLE_CONSTANTMEM
    gpuErrchk (cudaMemcpyToSymbol (api::maxtimers, &maxtimers_in, sizeof (int)));
    gpuErrchk (cudaMemcpyToSymbol (api::warpsize,  &warpsize_in,  sizeof (int)));
    gpuErrchk (cudaMemcpyToSymbol (api::maxwarps,  &maxwarps_in,  sizeof (int)));
#else
    int *dmaxtimers;
    int *dwarpsize;
    int *dmaxwarps;
    gpuErrchk (cudaGetSymbolAddress ((void **)&dmaxtimers, api::maxtimers));
    gpuErrchk (cudaGetSymbolAddress ((void **)&dwarpsize, api::warpsize));
    gpuErrchk (cudaGetSymbolAddress ((void **)&dmaxwarps, api::maxwarps));
    
    gpuErrchk (cudaMemcpy (dmaxtimers, &maxtimers_in, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy (dwarpsize,  &warpsize_in,  sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk (cudaMemcpy (dmaxwarps,  &maxwarps_in,  sizeof(int), cudaMemcpyHostToDevice));
#endif

    nbytes = maxtimers_in * sizeof (Timername);
    gpuErrchk (cudaMalloc (&timernames_cpu, nbytes));

    // Create space for a "return" value for __global__functions to be checked on CPU
    gpuErrchk (cudaMallocManaged (&retval, sizeof (int)));

#ifdef TIME_GPTL
    nbytes = maxwarps_in * NUM_INTERNAL_TIMERS * sizeof (long long);
    gpuErrchk (cudaMalloc (&globcount_cpu, nbytes));
#endif

    initialize_gpu <<<1,1>>> (verbose_in,
			      maxwarps_in,
			      gpu_hz_in,
			      timers_cpu,
			      timernames_cpu,
			      cores_per_sm_in,
			      globcount_cpu,
			      retval);
    cudaDeviceSynchronize ();
#ifdef ENABLE_CONSTANTMEM
    // Can only change a __constant__ value from host
    static bool truefalse;
    if (*retval == 0)
      truefalse = true;
    else
      truefalse = false;
    gpuErrchk (cudaMemcpyToSymbol (api::initialized, &truefalse, sizeof (bool)));
#endif
    // This should flush any existing print buffers
    cudaDeviceSynchronize ();
    return *retval;
  }

  /*
  ** GPTLfinalize_gpu: Close down the GPTL CUDA timers. Calls finalize_gpu <<<1,1>>>
  */
  __host__ int GPTLfinalize_gpu (void)
  {
    cudaError_t cudaret;
    static const char *thisfunc = "GPTLfinalize_gpu";
    // Create space for a "return" value for __global__functions to be checked on CPU
    if (retval == 0)        // address=0 means first call
      gpuErrchk (cudaMallocManaged (&retval, sizeof (int)));

    *retval = 0;  // Init to success, failure in the global routine will be non-zero
    finalize_gpu <<<1,1>>> (retval);
    cudaDeviceSynchronize ();

    cudaret = cudaGetLastError();
    if (cudaret != cudaSuccess) {
      printf("%s: %s\n", thisfunc, cudaGetErrorString(cudaret));
      return -1;
    }

    if (*retval != 0)
      printf ("GPTLfinalize_gpu: Failure from finalize_gpu\n");
    return *retval;
  }
}  // End of extern "C"

/*
** initialize_gpu (): Initialization routine must be called from single-threaded
**   region before any other timing routines may be called.  The need for this
**   routine could be eliminated if not targetting timing library for threaded
**   capability. 
*/
__global__ static void initialize_gpu (const int verbose_in,
				       const int maxwarps_in,
				       const double gpu_hz_in,
				       Timer *timers_cpu,
				       Timername *timernames_cpu,
				       const int cores_per_sm_in,
				       long long *globcount_cpu,
				       int *retval)
{
  int w, wi;        // warp, flattened indices
  long long t1, t2; // returned from underlying timer
  static const char *thisfunc = "initialize_gpu";

  *retval = 0;
#ifdef VERBOSE
  printf ("Entered %s\n", thisfunc);
#endif
  if (api::initialized) {
    (void) util::error_1s ("%s: has already been called\n", thisfunc);
    *retval = -1;
    return;
  }

  // Set global vars from input args
  api::verbose      = verbose_in;
  api::gpu_hz       = gpu_hz_in;
  api::warps_per_sm = cores_per_sm_in / api::warpsize;
  api::timers       = timers_cpu;
  api::timernames   = timernames_cpu;
#ifdef TIME_GPTL
  timingohd::globcount  = globcount_cpu;
  memset (timingohd::globcount, 0, api::maxwarps * NUM_INTERNAL_TIMERS * sizeof (long long));
#endif

  // Initialize timers
  api::ntimers = 0;
  api::max_name_len = 0;
  for (w = 0; w < api::maxwarps; ++w) {
    for (int i = 0; i < api::maxtimers; ++i) {
      wi = FLATTEN_TIMERS(w,i);
      memset (&api::timers[wi], 0, sizeof (Timer));
      api::timers[wi].wall.min = LLONG_MAX;
    }
  }
  // Make a timer "GPTL_ROOT" to ensure no orphans, and to simplify printing.
  memcpy (api::timernames[0].name, "GPTL_ROOT", 9+1);

  if (api::verbose) {
    t1 = clock64 ();
    t2 = clock64 ();
    if (t1 > t2)
      printf ("GPTL %s: negative delta-t=%lld\n", thisfunc, t2-t1);

    printf ("Per call overhead est. t2-t1=%g should be near zero\n", (float) (t2-t1));
    printf ("Underlying wallclock timing routine is clock64\n");
  }
#ifndef ENABLE_CONSTANTMEM
  // If "initialized" is __constant__ its value will be changed in the CPU caller
  api::initialized = true;
#endif
}

/*
** finalize_gpu (): Finalization routine must be called from single-threaded
**   region. Free all malloc'd space
**
** Output argument: Currently hard-wired to success, but could pass back an error condition
*/
__global__ void finalize_gpu (int *retval)
{
  static const char *thisfunc = "finalize_gpu";

  if ( ! api::initialized) {
    (void) util::error_1s ("%s: initialization was not completed\n", thisfunc);
    return;
  }

  free (api::timers);
  free (api::timernames);
  
  util::reset_errors_gpu ();

  // Reset initial values
  api::timers = 0;
  api::timernames = 0;
  api::max_name_len = 0;
#ifndef ENABLE_CONSTANTMEM
  // If "initialized" is __constant__ its value will be changed in the CPU caller
  api::initialized = false;
#endif
  api::verbose = false;
  *retval = 0;
}
