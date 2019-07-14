#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include "private.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

__host__ void GPTLprint_gpustats (FILE *fp, int maxwarps, int maxtimers, double gpu_hz, int devnum)
{
  Gpustats *gpustats;
  int *max_name_len_gpu;
  int *ngputimers;
  int extraspace;
  int i, n;
  int ret;
  int *maxwarpid_found;
  int *maxwarpid_timed;

  // Returned from GPTLget_overhead_gpu:
  long long *get_warp_num_ohdgpu; // Getting my thread index
  long long *startstop_ohdgpu;    // Cost est of start/stop pair
  long long *utr_ohdgpu;          // Underlying timing routine
  long long *start_misc_ohdgpu;   // misc code from GPTLstart_gpu
  long long *stop_misc_ohdgpu;    // misc code from GPTLstop_gpu
  long long *self_ohdgpu;         // Cost est. for timing this region
  long long *parent_ohdgpu;       // Cost est. to parent of this region
  long long *my_strlen_ohdgpu;    // my_strlen function
  long long *STRMATCH_ohdgpu;     // my_strcmp function
  // Returned from GPTLget_memstats_gpu:
  float *regionmem, *timernamemem;

  int count_max, count_min;
  double wallmax, wallmin;
  double self, parent;
  double gwn;
  double utr;
  double startstop;
  double tot;
  double startmisc, stopmisc;
#ifdef HAVE_MPI
  int myrank = 0;
  int mpi_active;
#endif

#define HOSTSIZE 32
  char hostname[HOSTSIZE];
  static const char *thisfunc = "GPTLprint_gpustats";

  gpuErrchk (cudaMallocManaged (&ngputimers,                   sizeof (int)));
  gpuErrchk (cudaMallocManaged (&max_name_len_gpu,             sizeof (int)));
  gpuErrchk (cudaMallocManaged (&gpustats,         maxtimers * sizeof (Gpustats)));

  gpuErrchk (cudaMallocManaged (&maxwarpid_found,     sizeof (int)));
  gpuErrchk (cudaMallocManaged (&maxwarpid_timed,     sizeof (int)));

  gpuErrchk (cudaMallocManaged (&get_warp_num_ohdgpu, sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&startstop_ohdgpu,    sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&utr_ohdgpu,          sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&start_misc_ohdgpu,   sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&stop_misc_ohdgpu,    sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&self_ohdgpu,         sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&parent_ohdgpu,       sizeof (long long)));

  gpuErrchk (cudaMallocManaged (&my_strlen_ohdgpu,    sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&STRMATCH_ohdgpu,     sizeof (long long)));

  gpuErrchk (cudaMallocManaged (&regionmem,           sizeof (float)));
  gpuErrchk (cudaMallocManaged (&timernamemem,        sizeof (float)));

  GPTLfill_gpustats<<<1,1>>> (gpustats, max_name_len_gpu, ngputimers);
  if (cudaGetLastError() != cudaSuccess)
    printf( "%s: Error from GPTLfill_gpustats\n", thisfunc);
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

  fprintf (fp, "\n\nGPU Results:\n");

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

  GPTLget_gpusizes <<<1,1>>> (maxwarpid_found, maxwarpid_timed);
  GPTLget_overhead_gpu <<<1,1>>> (get_warp_num_ohdgpu,
				  startstop_ohdgpu,
				  utr_ohdgpu,
				  start_misc_ohdgpu,
				  stop_misc_ohdgpu,
				  self_ohdgpu,
				  parent_ohdgpu,
				  my_strlen_ohdgpu,
				  STRMATCH_ohdgpu);
  cudaDeviceSynchronize();

  fprintf (fp, "Underlying timing routine was clock64() assumed @ %f Ghz\n", gpu_hz * 1.e-9);
  startstop = startstop_ohdgpu[0] / gpu_hz;
  fprintf (fp, "Total overhead of 1 GPTLstart_gpu + GPTLstop_gpu pair call=%7.1e seconds\n", startstop);
  fprintf (fp, "Components of the pair are as follows (Fortran layer ignored):\n");
  fprintf (fp, "NOTE: sum of overheads should be near start+stop but not necessarily exact\n");
  fprintf (fp, "This is because start/stop timing est. is done separately from components\n");

  gwn       = 2.*get_warp_num_ohdgpu[0] / gpu_hz;  // 2. is due to calls from both start and stop
  utr       = 2.*utr_ohdgpu[0] / gpu_hz;           // 2. is due to calls from both start and stop
  startmisc = start_misc_ohdgpu[0] / gpu_hz;
  stopmisc  = stop_misc_ohdgpu[0] / gpu_hz;
  tot       = gwn + utr + startmisc + stopmisc;

  fprintf (fp, "Get warp number:                %7.1e = %5.1f%% of total\n", gwn, 100.*(gwn/tot));
  fprintf (fp, "Underlying timing routine+SMID: %7.1e = %5.1f%% of total\n", utr, 100.*(utr/tot));
  fprintf (fp, "Misc calcs in GPTL_start_gpu:   %7.1e = %5.1f%% of total\n",
	   startmisc, 100.*(startmisc/tot));
  fprintf (fp, "Misc calcs in GPTL_stop_gpu:    %7.1e = %5.1f%% of total\n",
	   stopmisc, 100.*(stopmisc/tot));
  fprintf (fp, "\n");

  fprintf (fp, "These 2 are called only by GPTLinit_handle_gpu, thus not part of overhead:\n");
  fprintf (fp, "my_strlen:                      %7.1e (name=GPTL_ROOT)\n",
	   my_strlen_ohdgpu[0] / gpu_hz);
  fprintf (fp, "STRMATCH:                       %7.1e (matched name=GPTL_ROOT)\n",
	   STRMATCH_ohdgpu[0] / gpu_hz);
  fprintf (fp, "\n");

  printf ("%s: calling gpu kernel GPTLfill_gpustats...\n", thisfunc);
  printf ("%s: returned from GPTLfill_gpustats: printing results\n", thisfunc);

  fprintf (fp, "\nGPU timing stats\n");
  fprintf (fp, "GPTL could handle up to %d warps (%d threads)\n", maxwarps, maxwarps * WARPSIZE);
  fprintf (fp, "This setting can be changed with: GPTLsetoption(GPTLmaxthreads_gpu,<number>)\n");
  fprintf (fp, "%d = max warpId found\n", maxwarpid_found[0]);
  fprintf (fp, "%d = max warpId examined\n", maxwarpid_timed[0]);
  fprintf (fp, "Only warps which were timed are counted in the following stats\n");
  fprintf (fp, "Overhead estimates self_OH and parent_OH are for warp with \'maxcount\' calls\n");
  fprintf (fp, "Assuming SMs are always busy computing, GPTL overhead can be vaguely estimated by this calculation:\n");
  fprintf (fp, "(num warps allocated / num warps on device) * (self_OH + parent_OH)\n");

  fprintf (fp, "name            = region name\n");
  fprintf (fp, "calls           = number of invocations across all examined warps\n");
  fprintf (fp, "warps           = number of examined warps for which region was timed at least once\n");
  fprintf (fp, "holes           = number of examined warps for which region was never executed (maxwarpid_timed + 1 - nwarps(region)\n");
  fprintf (fp, "wallmax (warp)  = max wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "wallmin (warp)  = min wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "maxcount (warp) = max number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "mincount (warp) = min number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "negmax (warp)   = if a region had a negative interval, biggest count is printed along with the warp number responsible\n");
  fprintf (fp, "nwarps          = number of warps encountering a negative interval\n");
  fprintf (fp, "Bad_SM          = number of times smid changed (these instances are NOT timed!) Max possible = 'calls'\n");
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
    fprintf (fp, "%6d ", maxwarpid_timed[0] - gpustats[n].nwarps + 1); // number of (untimed) holes
    
    wallmax = gpustats[n].accum_max / gpu_hz;            // max time for name across warps
    if (wallmax < 0.01)
      fprintf (fp, "|%8.2e ", wallmax);
    else
      fprintf (fp, "|%8.3f ", wallmax);
    fprintf (fp, "%6d ",gpustats[n].accum_max_warp);     // warp number for max
    
    wallmin = gpustats[n].accum_min / gpu_hz;            // min time for name across warps
    if (wallmin < 0.01)
      fprintf (fp, "|%8.2e ", wallmin);
    else
      fprintf (fp, "|%8.3f ", wallmin);	       
    fprintf (fp, "%6d ",gpustats[n].accum_min_warp);     // warp number for min
    
    count_max = gpustats[n].count_max;
    if (count_max < PRTHRESH)
      fprintf (fp, "|%8d ", count_max);                   // max count for region "name"
    else
      fprintf (fp, "|%8.1e ", (float) count_max);
    fprintf (fp, "%6d ",gpustats[n].count_max_warp);     // warp which accounted for max times
    
    count_min = gpustats[n].count_min;                
    if (count_min < PRTHRESH)
      fprintf (fp, "|%8d ", count_min);                   // min count for region "name"
    else
      fprintf (fp, "|%8.1e ", (float) count_min);
    fprintf (fp, "%6d ",gpustats[n].count_min_warp);     // warp which accounted for max times

    if (gpustats[n].negdelta_count_max == 0) {
      fprintf (fp, "|    -    ");
      fprintf (fp, "   -   ");
      fprintf (fp, "   -   ");
    } else {
      fprintf (fp, "|%8d ", gpustats[n].negdelta_count_max);      // max negcount for "stop" region "name"
      fprintf (fp, "%6d ", gpustats[n].negdelta_count_max_warp); // warp which accounted for negdelta_count_max
      fprintf (fp, "%6d ", gpustats[n].negdelta_nwarps);         // number of warps which had > 0 negatives
    }

    if (gpustats[n].badsmid_count == 0)
      fprintf (fp, "|   -   |");
    else
      fprintf (fp, "|%6d |", gpustats[n].badsmid_count);      // number of times SM changed on "stop" call

    self = (gpustats[n].count_max * self_ohdgpu[0]) / gpu_hz; // self ohd est
    if (self < 0.01)
      fprintf (fp, "%8.2e  ", self);
    else
      fprintf (fp, "%8.3f  ", self);	       
    
    parent = (gpustats[n].count_max * parent_ohdgpu[0]) / gpu_hz; // parent ohd est
    if (self < 0.01)
      fprintf (fp, "%8.2e ", parent);
    else
      fprintf (fp, "%8.3f ", parent);	       

    fprintf (fp, "\n");
  }

  printf ("%s: calling gpu kernel GPTLget_memstats_gpu...\n", thisfunc);
  GPTLget_memstats_gpu <<<1,1>>> (regionmem, timernamemem);
  cudaDeviceSynchronize();
  fprintf (fp, "\n");
  fprintf (fp, "GPTL GPU memory usage (Timers)      = %8g KB\n", regionmem[0]*.001);
  fprintf (fp, "GPTL GPU memory usage (Timer names) = %8g KB\n", timernamemem[0]*.001);
  fprintf (fp, "                                      --------\n");
  fprintf (fp, "GPTL GPU memory usage (Total)       = %8g KB\n", (regionmem[0] + timernamemem[0])*.001);
}
