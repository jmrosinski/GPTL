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
  long long *ftn_ohdgpu;            // Fortran wrapper overhead
  long long *get_thread_num_ohdgpu; /* Getting my thread index */
  long long *genhashidx_ohdgpu;     /* Generating hash index */
  long long *getentry_ohdgpu;       /* Finding entry in hash table */
  char *getentry_ohdgpu_name; // name used for getentry test
  long long *utr_ohdgpu;            /* Underlying timing routine */
  long long *self_ohdgpu;           // Cost est. for timing this region
  long long *parent_ohdgpu;         // Cost est. to parent of this region

  long long *my_strlen_ohdgpu;
  long long *STRMATCH_ohdgpu;
  // Returned from GPTLget_memstats_gpu:
  float *hashmem;
  float *regionmem;

  int count_max, count_min;
  double wallmax, wallmin;
  double self, parent;
  double tot_ohdgpu;
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

  gpuErrchk (cudaMallocManaged (&maxwarpid_found,              sizeof (int)));
  gpuErrchk (cudaMallocManaged (&maxwarpid_timed,              sizeof (int)));

  gpuErrchk (cudaMallocManaged (&ftn_ohdgpu,            sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&get_thread_num_ohdgpu, sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&genhashidx_ohdgpu,     sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&getentry_ohdgpu,       sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&getentry_ohdgpu_name,  MAX_CHARS+1));
  gpuErrchk (cudaMallocManaged (&utr_ohdgpu,            sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&self_ohdgpu,           sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&parent_ohdgpu,         sizeof (long long)));

  gpuErrchk (cudaMallocManaged (&my_strlen_ohdgpu,      sizeof (long long)));
  gpuErrchk (cudaMallocManaged (&STRMATCH_ohdgpu,       sizeof (long long)));

  gpuErrchk (cudaMallocManaged (&hashmem,               sizeof (float)));
  gpuErrchk (cudaMallocManaged (&regionmem,             sizeof (float)));

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
  GPTLget_overhead_gpu <<<1,1>>> (ftn_ohdgpu,
				  get_thread_num_ohdgpu,
				  genhashidx_ohdgpu,
				  getentry_ohdgpu,
				  getentry_ohdgpu_name,
				  utr_ohdgpu,
				  self_ohdgpu,
				  parent_ohdgpu,
				  my_strlen_ohdgpu,
				  STRMATCH_ohdgpu);
  cudaDeviceSynchronize();

  fprintf (fp, "Underlying timing routine was clock64() assumed @ %f Ghz\n", gpu_hz * 1.e-9);
  tot_ohdgpu = (ftn_ohdgpu[0] + get_thread_num_ohdgpu[0] + genhashidx_ohdgpu[0] + 
		getentry_ohdgpu[0] + utr_ohdgpu[0]) / gpu_hz;
  fprintf (fp, "Total overhead of 1 GPTLstart_gpu or GPTLstop_gpu call=%g seconds\n", tot_ohdgpu);
  fprintf (fp, "Components are as follows:\n");
  fprintf (fp, "Fortran layer:                  %7.1e = %5.1f%% of total\n", 
	   ftn_ohdgpu[0] / gpu_hz, ftn_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Get thread number:              %7.1e = %5.1f%% of total\n", 
	   get_thread_num_ohdgpu[0] / gpu_hz, get_thread_num_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Generate hash index:            %7.1e = %5.1f%% of total\n", 
	   genhashidx_ohdgpu[0] / gpu_hz, genhashidx_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Find hashtable entry:           %7.1e = %5.1f%% of total (name=%s)\n", 
	   getentry_ohdgpu[0] / gpu_hz, getentry_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz), getentry_ohdgpu_name );
  fprintf (fp, "Underlying timing routine:      %7.1e = %5.1f%% of total\n", 
	   utr_ohdgpu[0] / gpu_hz, utr_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "\n");

  fprintf (fp, "my_strlen (part of genhashidx): %7.1e\n", my_strlen_ohdgpu[0] / gpu_hz);
  fprintf (fp, "STRMATCH (part of getentry):    %7.1e\n", STRMATCH_ohdgpu[0] / gpu_hz);
  fprintf (fp, "\n");

  printf ("%s: calling gpu kernel GPTLfill_gpustats...\n", thisfunc);
  printf ("%s: returned from GPTLfill_gpustats: printing results\n", thisfunc);

  fprintf (fp, "\nGPU timing stats\n");
  fprintf (fp, "GPTL could handle up to %d warps (%d threads)\n", maxwarps, maxwarps * WARPSIZE);
  fprintf (fp, "This setting can be changed with: GPTLsetoption(GPTLmaxthreads_gpu,<number>)\n");
  fprintf (fp, "%d = max warpId found\n", maxwarpid_found[0]);
  fprintf (fp, "%d = max warpId timed\n", maxwarpid_timed[0]);
  fprintf (fp, "Only warps which were timed are counted in the following stats\n");
  fprintf (fp, "Overhead estimates self_OH and parent_OH are for warp with \'maxcount\' calls\n");
  fprintf (fp, "OHD estimate assumes Fortran, and non-handle routines used\n");
  fprintf (fp, "Actual overhead can be reduced by using \'handle\' routines and \'_c\' Fortran routines\n\n");

  fprintf (fp, "name            = region name\n");
  fprintf (fp, "calls           = number of invocations across all timed warps\n");
  fprintf (fp, "warps           = number of timed warps for which region was timed at least once\n");
  fprintf (fp, "holes           = number of timed warps for which region was never timed (maxwarpid_timed + 1 - nwarps(region)\n");
  fprintf (fp, "wallmax (warp)  = max wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "wallmin (warp)  = min wall time (sec) taken by any timed warp for this region, followed by the warp number\n");
  fprintf (fp, "maxcount (warp) = max number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "mincount (warp) = min number of times region invoked by any timed warp, followed by the warp number\n");
  fprintf (fp, "negmxstrt(warp) = if a start region had a neg intrvl, biggest count is printed along with the warp number responsible\n");
  fprintf (fp, "nwarps          = number of warps encountering negmxstrt > 0\n");
  fprintf (fp, "negmxstop(warp) = if a stop region had a neg intrvl, biggest count is printed along with the warp number responsible\n");
  fprintf (fp, "nwarps          = number of warps encountering negmxstop > 0\n");
#ifdef CHECK_SM
  fprintf (fp, "smstrt          = number of warps encountering change of smid on start\n");
  fprintf (fp, "smstop          = number of warps encountering change of smid on stop\n");
#endif
  fprintf (fp, "self_OH         = estimate of GPTL overhead (sec) in the timer incurred by 'maxcount' invocations of it\n");
  fprintf (fp, "parent_OH       = estimate of GPTL overhead (sec) in the parent of the timer incurred by 'maxcount' invocations of it\n\n");
  // Print header, padding to length of longest name
  extraspace = max_name_len_gpu[0] - 4; // "name" is 4 chars
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");
#ifdef CHECK_SM
  fprintf (fp, "name    calls  warps  holes  wallmax  (warp) wallmin (warp) maxcount (warp) mincount (warp) negmxstrt(warp) nwarps negmxstop(warp) nwarps   smstrt   smstop  self_OH parent_OH\n");
#else
  fprintf (fp, "name    calls  warps  holes  wallmax  (warp) wallmin (warp) maxcount (warp) mincount (warp) negmxstrt(warp) nwarps negmxstop(warp) nwarps  self_OH parent_OH\n");
#endif
  for (n = 0; n < ngputimers[0]; ++n) {
    extraspace = max_name_len_gpu[0] - strlen (gpustats[n].name);
    for (i = 0; i < extraspace; ++i)
      fprintf (fp, " ");
    fprintf (fp, "%s ", gpustats[n].name);               // region name
    if (gpustats[n].count < 1000000)
      fprintf (fp, "%8lu ", gpustats[n].count);           // # start/stops of region
    else
      fprintf (fp, "%8.2e ", (float) gpustats[n].count); // # start/stops of region  

    fprintf (fp, "%6d ", gpustats[n].nwarps);            // nwarps involving name
    fprintf (fp, "%6d ", maxwarpid_timed[0] - gpustats[n].nwarps + 1);            // number of (untimed) holes
    
    wallmax = gpustats[n].accum_max / gpu_hz;            // max time for name across warps
    if (wallmax < 0.01)
      fprintf (fp, "%8.2e ", wallmax);
    else
      fprintf (fp, "%8.3f ", wallmax);
    fprintf (fp, "%6d ",gpustats[n].accum_max_warp);     // warp number for max
    
    wallmin = gpustats[n].accum_min / gpu_hz;            // min time for name across warps
    if (wallmin < 0.01)
      fprintf (fp, "%8.2e ", wallmin);
    else
      fprintf (fp, "%8.3f ", wallmin);	       
    fprintf (fp, "%6d ",gpustats[n].accum_min_warp);   // warp number for min
    
    count_max = gpustats[n].count_max;
    if (count_max < PRTHRESH)
      fprintf (fp, "%8d ", count_max);                 // max count for region "name"
    else
      fprintf (fp, "%8.1e ", (float) count_max);
    fprintf (fp, "%6d ",gpustats[n].count_max_warp);   // warp which accounted for max times
    
    count_min = gpustats[n].count_min;                
    if (count_min < PRTHRESH)
      fprintf (fp, "%8d ", count_min);                  // min count for region "name"
    else
      fprintf (fp, "%8.1e ", (float) count_min);
    fprintf (fp, "%6d ",gpustats[n].count_min_warp);    // warp which accounted for max times

    fprintf (fp, "%8d ", gpustats[n].negcount_start_max);      // max negcount for "start" region "name" (hope it's zero)
    fprintf (fp, "%6d ", gpustats[n].negcount_start_max_warp); // warp which accounted for negcount_start_max
    fprintf (fp, "%6d ", gpustats[n].negstart_nwarps);         // number of warps which had > 0 negative starts

    fprintf (fp, "%8d ", gpustats[n].negcount_stop_max);       // max negcount for "stop" region "name" (hope it's zero)
    fprintf (fp, "%6d ", gpustats[n].negcount_stop_max_warp);  // warp which accounted for negcount_stop_max
    fprintf (fp, "%6d ", gpustats[n].negstop_nwarps);          // number of warps which had > 0 negative stops

#ifdef CHECK_SM
    if (gpustats[n].badsmid_start_count < PRTHRESH)
      fprintf (fp, "%8d ", gpustats[n].badsmid_start_count);     // number of times SM changed on "start" call
    else
      fprintf (fp, "%8.1e ", (float) gpustats[n].badsmid_start_count);

    if (gpustats[n].badsmid_stop_count < PRTHRESH)
      fprintf (fp, "%8d ", gpustats[n].badsmid_stop_count);      // number of times SM changed on "stop" call
    else
      fprintf (fp, "%8.1e ", (float) gpustats[n].badsmid_stop_count);
#endif

    self = gpustats[n].count_max * self_ohdgpu[0] / gpu_hz;    // self ohd est
    if (self < 0.01)
      fprintf (fp, "%8.2e  ", self);
    else
      fprintf (fp, "%8.3f  ", self);	       
    
    parent = gpustats[n].count_max * parent_ohdgpu[0] / gpu_hz; // parent ohd est
    if (self < 0.01)
      fprintf (fp, "%8.2e ", parent);
    else
      fprintf (fp, "%8.3f ", parent);	       

    fprintf (fp, "\n");
  }

  printf ("%s: calling gpu kernel GPTLget_memstats_gpu...\n", thisfunc);
  GPTLget_memstats_gpu <<<1,1>>> (hashmem, regionmem);
  cudaDeviceSynchronize();
  fprintf (fp, "\n");
  fprintf (fp, "Total GPTL GPU memory usage = %g KB\n", (hashmem[0] + regionmem[0])*.001);
  fprintf (fp, "Components:\n");
  fprintf (fp, "Hashmem                     = %g KB\n"
	       "Regionmem                   = %g KB\n", hashmem[0]*.001, regionmem[0]*.001);
}
