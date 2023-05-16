#include "config.h" // Must be first include.
// gptl gpu-private 
#include "output.h"
#include "api.h"
#include "util.h"
#include "overhead.h"
#ifdef TIME_GPTL
#include "timingohd.h"
#endif
// system
#include <stdio.h>
#include <unistd.h>
#include <limits.h>        // LLONG_MAX
#include <cuda.h>

namespace output {
  int *maxwarpid_found;  // for both CPU and GPU
  int *maxwarpid_timed;  // for both CPU and GPU
}

// GPTLprint_gpustats is called from gptl.c so wrap everything in extern "C"
extern "C" {
// Local function prototypes
__host__ static float getavg (float *, int);

// GPTLprint_gpustats: main routine gathers GPU stats and prints them
__host__ int GPTLprint_gpustats (FILE *fp, int warpsize_in, int warps_per_sm_in, int maxwarps_in,
				 int maxtimers_in, double gpu_hz_in, int devnum)
{
  Gpustats *gpustats;
  int *max_name_len_gpu;
  int *ngputimers;
  int extraspace;
  int i, n;
  int ret;
 
  // Returned from GPTLget_overhead_gpu:
  float *get_warp_num_ohdgpu; // Getting my thread index
  float *startstop_ohdgpu;    // Cost est of start/stop pair
  float *utr_ohdgpu;          // Underlying timing routine
  float *start_misc_ohdgpu;   // misc code from GPTLstart_gpu
  float *stop_misc_ohdgpu;    // misc code from GPTLstop_gpu
  float *self_ohdgpu;         // Cost est. for timing this region
  float *parent_ohdgpu;       // Cost est. to parent of this region
  float *my_strlen_ohdgpu;    // my_strlen function
  float *STRMATCH_ohdgpu;     // my_strcmp function
  // Returned from get_memstats_gpu:
  float *regionmem, *timernamemem;
  int *retval;                // return code from global functions

  int count_max, count_min;
  double wallmax, wallmin;
  double self, parent;
  double gwn;
  double utr;
  double startstop;
  double tot;
  double startmisc, stopmisc;
  double scalefac;
#ifdef HAVE_MPI
  int myrank = 0;
  int mpi_active;
#endif

#define HOSTSIZE 32
  char hostname[HOSTSIZE];
  static int already_called = 0; // Only call this routine once
  static const char *thisfunc = "GPTLprint_gpustats";

  if (already_called) {
    printf ("%s was already called. Cannot call again\n", thisfunc);
    return -1;
  } else {
    already_called = 1;
  }
  gpuErrchk (cudaMallocManaged (&ngputimers,                      sizeof (int)));
  gpuErrchk (cudaMallocManaged (&max_name_len_gpu,                sizeof (int)));
  gpuErrchk (cudaMallocManaged (&gpustats,         maxtimers_in * sizeof (Gpustats)));

  gpuErrchk (cudaMallocManaged (&output::maxwarpid_found,     sizeof (int)));
  gpuErrchk (cudaMallocManaged (&output::maxwarpid_timed,     sizeof (int)));

  gpuErrchk (cudaMallocManaged (&get_warp_num_ohdgpu, warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&startstop_ohdgpu,    warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&utr_ohdgpu,          warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&start_misc_ohdgpu,   warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&stop_misc_ohdgpu,    warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&self_ohdgpu,         warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&parent_ohdgpu,       warps_per_sm_in*sizeof (float)));

  gpuErrchk (cudaMallocManaged (&my_strlen_ohdgpu,    warps_per_sm_in*sizeof (float)));
  gpuErrchk (cudaMallocManaged (&STRMATCH_ohdgpu,     warps_per_sm_in*sizeof (float)));

  gpuErrchk (cudaMallocManaged (&regionmem,           sizeof (float)));
  gpuErrchk (cudaMallocManaged (&timernamemem,        sizeof (float)));

  // Create space for a "return" value for __global__functions to be checked on CPU
  gpuErrchk (cudaMallocManaged (&retval,       sizeof (int)));

  fprintf (fp, "--------------------------------------------------------------------------------\n");
  fprintf (fp, "\n\nGPU Results:\n");
  fprintf (fp, "GPU-specific GPTL build info:\n");

  fprintf (fp, "Compute capability was %d\n", CCAB);

#ifdef ENABLE_GPUCHECKS
  fprintf (fp, "ENABLE_GPUCHECKS was true\n");
#else
  fprintf (fp, "ENABLE_GPUCHECKS was false\n");
#endif

#ifdef ENABLE_CONSTANTMEM
  fprintf (fp, "ENABLE_CONSTANTMEM was true\n");
#else
  fprintf (fp, "ENABLE_CONSTANTMEM was false\n");
#endif

#ifdef ENABLE_GPURECURSION
  fprintf (fp, "ENABLE_GPURECURSION was true\n");
#else
  fprintf (fp, "ENABLE_GPURECURSION was false\n");
#endif

#ifdef TIME_GPTL
  fprintf (fp, "TIME_GPTL (on GPU) was true\n");
#else
  fprintf (fp, "TIME_GPTL (on GPU) was false\n");
#endif

#ifdef ENABLE_FOUND
  fprintf (fp, "ENABLE_FOUND (print max warpid found on GPU) was true\n");
#else
  fprintf (fp, "ENABLE_FOUND (print max warpid found on GPU) was false\n");
#endif

  fill_all_gpustats <<<1,1>>> (gpustats, max_name_len_gpu, ngputimers);
  if (cudaGetLastError() != cudaSuccess)
    printf( "%s: Error from fill_all_gpustats\n", thisfunc);
  cudaDeviceSynchronize();

#ifdef DEBUG_PRINT
  printf ("%s: ngputimers=%d\n",       thisfunc, *ngputimers);
  printf ("%s: max_name_len_gpu=%d\n", thisfunc, *max_name_len_gpu);
  for (n = 0; n < *ngputimers; ++n) {
    printf ("%s: timer=%s accum_max=%lld accum_min=%lld count_max=%d nwarps=%d\n", 
	    thisfunc, gpustats[n].name, gpustats[n].accum_max, gpustats[n].accum_min, 
	    gpustats[n].count_max, gpustats[n].nwarps);
  }
#endif

#ifdef HAVE_MPI
  ret = MPI_Initialized (&mpi_active);
  if (mpi_active) {
    ret = MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    fprintf (fp, "%s: MPI rank=%d\n", thisfunc, myrank);
  }
#endif
  fprintf (fp, "%s: device number=%d\n", thisfunc, devnum);
  ret = gethostname (hostname, HOSTSIZE);
  fprintf (fp, "%s: hostname=%s\n", thisfunc, hostname);

  util::get_maxwarpid_info <<<1,1>>> (output::maxwarpid_timed, output::maxwarpid_found);
  cudaDeviceSynchronize();

  // Reset timer GPTL_ROOT (index 0) so overhead tests succeed
  util::reset_gpu <<<1,1>>> (0, retval);
  cudaDeviceSynchronize();
  if (*retval != 0) {
    printf ("%s: Failure from reset_gpu(0). Cannot print GPU stats\n", thisfunc);
    return *retval;
  }

  // Initialize retval to success. If any thread encounters a problem they will
  // reset it to -1
  *retval = 0;
  // Can change to <<<1,1>>> if problems occur, but accuracy of overhead
  // estimates will be compromised.
  overhead::get_overhead_gpu <<<warps_per_sm_in,1>>> (get_warp_num_ohdgpu,
						      startstop_ohdgpu,
						      utr_ohdgpu,
						      start_misc_ohdgpu,
						      stop_misc_ohdgpu,
						      self_ohdgpu,
						      parent_ohdgpu,
						      my_strlen_ohdgpu,
						      STRMATCH_ohdgpu,
						      retval);
  cudaDeviceSynchronize();
  if (*retval != 0) {
    printf ("%s: Failure from GPTLget_overhead_gpu. Cannot print GPU stats\n", thisfunc);
    return *retval;
  }
  
  // 2. in next 2 computations is due to calls from both start and stop  
  gwn       = 2.*getavg (get_warp_num_ohdgpu, warps_per_sm_in) / gpu_hz_in;  
  utr       = 2.*getavg (utr_ohdgpu, warps_per_sm_in) / gpu_hz_in;
  startmisc = getavg (start_misc_ohdgpu, warps_per_sm_in) / gpu_hz_in;
  stopmisc  = getavg (stop_misc_ohdgpu, warps_per_sm_in) / gpu_hz_in;
  tot       = gwn + utr + startmisc + stopmisc;

  startstop = getavg (startstop_ohdgpu, warps_per_sm_in) / gpu_hz_in;
  scalefac  = startstop / tot;

  fprintf (fp, "Underlying timing routine was clock64() assumed @ %f Ghz\n", gpu_hz_in * 1.e-9);
  fprintf (fp, "Total overhead of 1 GPTLstart_gpu + GPTLstop_gpu pair call=%7.1e seconds\n", startstop);
  fprintf (fp, "Components of the pair are as follows:\n");
  fprintf (fp, "Sum of overheads should be near start+stop but not necessarily exact (scalefac = %6.2f)\n",
	   scalefac);
  fprintf (fp, "This is because start/stop timing est. is done separately from components\n");
  fprintf (fp, "Get warp number:                %7.1e = %5.1f%% of total\n", gwn, 100.*(gwn/tot));
  fprintf (fp, "Underlying timing routine+SMID: %7.1e = %5.1f%% of total\n", utr, 100.*(utr/tot));
  fprintf (fp, "Misc calcs in GPTLstart_gpu:    %7.1e = %5.1f%% of total\n",
	   startmisc, 100.*(startmisc/tot));
  fprintf (fp, "Misc calcs in GPTLstop_gpu:     %7.1e = %5.1f%% of total\n",
	   stopmisc, 100.*(stopmisc/tot));
  fprintf (fp, "\n");

  fprintf (fp, "These 2 are called only by GPTLinit_handle_gpu, thus not part of overhead:\n");
  fprintf (fp, "my_strlen:                      %7.1e (name=GPTL_ROOT)\n",
	   getavg (my_strlen_ohdgpu, warps_per_sm_in) / gpu_hz_in);
  fprintf (fp, "STRMATCH:                       %7.1e (matched name=GPTL_ROOT)\n",
	   getavg (STRMATCH_ohdgpu, warps_per_sm_in) / gpu_hz_in);
  fprintf (fp, "\n");

  fprintf (fp, "\nGPU timing stats\n");
  fprintf (fp, "GPTL could handle up to %d warps (%d threads)\n",
	   maxwarps, maxwarps_in * warpsize_in);
  fprintf (fp, "This setting can be changed with: GPTLsetoption(GPTLmaxthreads_gpu,<number>)\n");
  fprintf (fp, "%d = max warpId examined\n", output::maxwarpid_timed[0]);
#ifdef ENABLE_FOUND
  fprintf (fp, "%d = ESTIMATE of max warpId found. Could be bigger caused by race condition\n",
	   output::maxwarpid_found[0]);
#endif
  fprintf (fp, "Only warps which were timed are counted in the following stats\n");
  fprintf (fp, "Overhead estimates self_OH and parent_OH are for warp with \'maxcount\' calls\n");
  fprintf (fp, "Assuming SMs are always busy computing, GPTL overhead can be vaguely estimated"
	   "by this calculation:\n");
  fprintf (fp, "(num warps allocated / num warps on device) * (self_OH + parent_OH)\n");

  fprintf (fp, "name            = region name\n");
  fprintf (fp, "calls           = number of invocations across all examined warps\n");
  fprintf (fp, "warps           = number of examined warps for which region was timed at least once\n");
  fprintf (fp, "holes           = number of examined warps for which region was never executed (maxwarpid_timed + 1 - num_warps_timed\n");
  fprintf (fp, "wallmax (warp)  = max wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "wallmin (warp)  = min wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "maxcount (warp) = max number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "mincount (warp) = min number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "negmax (warp)   = if a region had a negative interval, biggest count is printed along with the warp number responsible\n");
  fprintf (fp, "nwarps          = number of warps encountering a negative interval\n");
  fprintf (fp, "Bad_SM          = number of times smid changed Max possible = 'calls.' Reported time could be WILDLY wrong\n");
  fprintf (fp, "self_OH         = estimate of GPTL overhead (sec) in the timer incurred by 'maxcount' invocations of it\n");
  fprintf (fp, "parent_OH       = estimate of GPTL overhead (sec) in the parent of the timer incurred by 'maxcount' invocations of it\n\n");
  // Print header, padding to length of longest name
  extraspace = max_name_len_gpu[0] - 4; // "name" is 4 chars
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");
  fprintf (fp, "name    calls  warps  holes | wallmax  (warp)| wallmin (warp) | maxcount (warp)| mincount (warp)| negmax (warp) nwarps  | Bad_SM| self_OH parent_OH\n");
  for (n = 0; n < ngputimers[0]; ++n) {
    extraspace = max_name_len_gpu[0] - strlen (gpustats[n].name);
    for (i = 0; i < extraspace; ++i)
      fprintf (fp, " ");
    fprintf (fp, "%s ", gpustats[n].name);               // region name
    if (gpustats[n].count < 1000000)
      fprintf (fp, "%8lu ", gpustats[n].count);          // # start/stops of region
    else
      fprintf (fp, "%8.2e ", (float) gpustats[n].count); // # start/stops of region  

    fprintf (fp, "%6d ", gpustats[n].nwarps);            // nwarps involving name
    fprintf (fp, "%6d ", output::maxwarpid_timed[0] - gpustats[n].nwarps + 1); // number of (untimed) holes
    
    wallmax = gpustats[n].accum_max / gpu_hz_in;            // max time for name across warps
    if (wallmax < 0.01)
      fprintf (fp, "|%8.2e ", wallmax);
    else
      fprintf (fp, "|%8.3f ", wallmax);
    fprintf (fp, "%6d ",gpustats[n].accum_max_warp);     // warp number for max
    
    wallmin = gpustats[n].accum_min / gpu_hz_in;            // min time for name across warps
    if (wallmin < 0.01)
      fprintf (fp, "|%8.2e ", wallmin);
    else
      fprintf (fp, "|%8.3f ", wallmin);	       
    fprintf (fp, "%6d ",gpustats[n].accum_min_warp);     // warp number for min
    
    count_max = gpustats[n].count_max;
    if (count_max < PRTHRESH)
      fprintf (fp, "|%8d ", count_max);                  // max count for region "name"
    else
      fprintf (fp, "|%8.1e ", (float) count_max);
    fprintf (fp, "%6d ",gpustats[n].count_max_warp);     // warp which accounted for max times
    
    count_min = gpustats[n].count_min;                
    if (count_min < PRTHRESH)
      fprintf (fp, "|%8d ", count_min);                  // min count for region "name"
    else
      fprintf (fp, "|%8.1e ", (float) count_min);
    fprintf (fp, "%6d ",gpustats[n].count_min_warp);     // warp which accounted for max times

    if (gpustats[n].negdelta_count_max == 0) {
      fprintf (fp, "|    -    ");
      fprintf (fp, "   -   ");
      fprintf (fp, "   -   ");
    } else {
      fprintf (fp, "|%8d ", gpustats[n].negdelta_count_max);
      fprintf (fp, "%6d ", gpustats[n].negdelta_count_max_warp);
      fprintf (fp, "%6d ", gpustats[n].negdelta_nwarps);
    }

    if (gpustats[n].badsmid_count == 0)
      fprintf (fp, "|   -   |");
    else
      fprintf (fp, "|%6d |", gpustats[n].badsmid_count);
    // self ohd est
    self = (gpustats[n].count_max * getavg (self_ohdgpu, warps_per_sm_in)) / gpu_hz_in;
    self *= scalefac;   // try to get a closer estimate
    if (self < 0.01)
      fprintf (fp, "%8.2e  ", self);
    else
      fprintf (fp, "%8.3f  ", self);	       
    
    parent = (gpustats[n].count_max * getavg (parent_ohdgpu, warps_per_sm_in)) / gpu_hz_in; // parent ohd est
    parent *= scalefac;                                           // try to get a closer estimate
    if (self < 0.01)
      fprintf (fp, "%8.2e ", parent);
    else
      fprintf (fp, "%8.3f ", parent);	       

    fprintf (fp, "\n");
  }

  overhead::get_memstats_gpu <<<1,1>>> (regionmem, timernamemem);
  cudaDeviceSynchronize();
  fprintf (fp, "\n");
  fprintf (fp, "GPTL GPU memory usage (Timers)      = %8g KB\n", regionmem[0]*.001);
  fprintf (fp, "GPTL GPU memory usage (Timer names) = %8g KB\n", timernamemem[0]*.001);
  fprintf (fp, "                                      --------\n");
  fprintf (fp, "GPTL GPU memory usage (Total)       = %8g KB\n", (regionmem[0] + timernamemem[0])*.001);
  return 0;
}

// getavg: compute and average over number of warps
__host__ float getavg (float *arr, int warps_per_sm_in)
{
  float avg = 0.;
  for (int i = 0; i < warps_per_sm_in; ++i)
    avg += arr[i];
  avg /= warps_per_sm_in;
  return avg;
}
}
