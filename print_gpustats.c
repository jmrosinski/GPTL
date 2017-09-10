#include <stdio.h>
#include <unistd.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "private.h"

extern int GPTLget_gpusizes (int [], int []);
extern int GPTLget_overhead_gpu (long long [],            /* Fortran overhead */
				 long long [],            /* Getting my thread index */
				 long long [],            /* Generating hash index */
				 long long [],            /* Finding entry in hash table */
				 long long [],            /* Underlying timing routine */
				 long long [],            /* self_ohd */
				 long long []);           /* parent_ohd */
extern int GPTLfill_gpustats (Gpustats [], int [], int []);

#pragma acc routine (GPTLget_gpusizes) seq
#pragma acc routine (GPTLfill_gpustats) seq 
#pragma acc routine (GPTLget_memstats_gpu) seq
#pragma acc routine (GPTLget_overhead_gpu) seq

void GPTLprint_gpustats (FILE *fp, double gpu_hz, int maxthreads_gpu, int devnum)
{
  //JR Use arrays of length 1 not scalars so "acc copyout" works properly!
  int nwarps_found[1];
  int nwarps_timed[1];
  int max_name_len_gpu[1];
  int ngputimers[1];

  // Returned from GPTLget_overhead_gpu:
  long long ftn_ohdgpu[1];            // Fortran wrapper overhead
  long long get_thread_num_ohdgpu[1]; /* Getting my thread index */
  long long genhashidx_ohdgpu[1];     /* Generating hash index */
  long long getentry_ohdgpu[1];       /* Finding entry in hash table */
  long long utr_ohdgpu[1];            /* Underlying timing routine */
  long long self_ohdgpu[1];           // Cost est. for timing this region
  long long parent_ohdgpu[1];         // Cost est. to parent of this region

  // Returned from GPTLget_memstats_gpu:
  float hashmem[1];
  float regionmem[1];

  int count_max, count_min;
  int extraspace;
  int i, n;
  int ret;
  double wallmax, wallmin;
  double self, parent;
  double tot_ohdgpu;
  Gpustats gpustats[MAX_GPUTIMERS];
  int myrank = 0;
  int mpi_active;
#define HOSTSIZE 32
  char hostname[HOSTSIZE];
  static const char *thisfunc = "GPTLprint_gpustats";

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

#pragma acc kernels copyout(ret, nwarps_found, nwarps_timed)
  ret = GPTLget_gpusizes (nwarps_found, nwarps_timed);

#pragma acc kernels copyout(ret, ftn_ohdgpu, get_thread_num_ohdgpu, genhashidx_ohdgpu, \
			    getentry_ohdgpu, utr_ohdgpu, self_ohdgpu, parent_ohdgpu)			     
  ret = GPTLget_overhead_gpu (ftn_ohdgpu,
			      get_thread_num_ohdgpu,
			      genhashidx_ohdgpu,
			      getentry_ohdgpu,
			      utr_ohdgpu,
			      self_ohdgpu,
			      parent_ohdgpu);

  fprintf (fp, "Underlying timing routine was clock64()\n");
  tot_ohdgpu = (ftn_ohdgpu[0] + get_thread_num_ohdgpu[0] + genhashidx_ohdgpu[0] + 
		getentry_ohdgpu[0] + utr_ohdgpu[0]) / gpu_hz;
  fprintf (fp, "Total overhead of 1 GPTLstart_gpu or GPTLstop_gpu call=%g seconds\n", tot_ohdgpu);
  fprintf (fp, "Components are as follows:\n");
  fprintf (fp, "Fortran layer:             %7.1e = %5.1f%% of total\n", 
	   ftn_ohdgpu[0] / gpu_hz, ftn_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Get thread number:         %7.1e = %5.1f%% of total\n", 
	   get_thread_num_ohdgpu[0] / gpu_hz, get_thread_num_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Generate hash index:       %7.1e = %5.1f%% of total\n", 
	   genhashidx_ohdgpu[0] / gpu_hz, genhashidx_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Find hashtable entry:      %7.1e = %5.1f%% of total\n", 
	   getentry_ohdgpu[0] / gpu_hz, getentry_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );
  fprintf (fp, "Underlying timing routine: %7.1e = %5.1f%% of total\n", 
	   utr_ohdgpu[0] / gpu_hz, utr_ohdgpu[0] * 100. / (tot_ohdgpu * gpu_hz) );

  printf ("%s: calling gpu kernel GPTLfill_gpustats...\n", thisfunc);
#pragma acc kernels copyout(ret, gpustats, max_name_len_gpu, ngputimers)
  ret = GPTLfill_gpustats (gpustats, max_name_len_gpu, ngputimers);
  printf ("%s: returned from GPTLfill_gpustats: printing results\n", thisfunc);

  fprintf (fp, "\nGPU timing stats\n");
  fprintf (fp, "GPTL could handle up to %d warps (%d threads)\n", 
	   maxthreads_gpu / WARPSIZE, maxthreads_gpu);
  fprintf (fp, "This setting can be changed with: GPTLsetoption(GPTLmaxthreads_gpu,<number>)\n");
  fprintf (fp, "%d warps were found\n", nwarps_found[0]);
  fprintf (fp, "%d warps were timed\n", nwarps_timed[0]);
  fprintf (fp, "Only warps which were timed are counted in the following stats\n");
  fprintf (fp, "Overhead estimates self_OH and parent_OH are for warp with \'maxcount\' calls\n");
  fprintf (fp, "OHD estimate assumes Fortran, and non-handle routines used\n");
  fprintf (fp, "Actual overhead can be reduced by using \'handle\' routines and \'_c\' Fortran routines\n");
  // Print header, padding to length of longest name
  extraspace = max_name_len_gpu[0] - 4; // "name" is 4 chars
  for (i = 0; i < extraspace; ++i)
    fprintf (fp, " ");
  fprintf (fp, "name calls warps  wallmax (warp) wallmin (warp) maxcount (warp) mincount (warp) self_OH parent_OH\n");
  for (n = 0; n < ngputimers[0]; ++n) {
    extraspace = max_name_len_gpu[0] - strlen (gpustats[n].name);
    for (i = 0; i < extraspace; ++i)
      fprintf (fp, " ");
    fprintf (fp, "%s ", gpustats[n].name);             // regopm name
    fprintf (fp, "%5d ", gpustats[n].count);           // # start/stops of region 
    fprintf (fp, "%5d ", gpustats[n].nwarps);          // nwarps_timed involving name
    
    wallmax = gpustats[n].accum_max / gpu_hz;          // max time for name across warps
    if (wallmax < 0.01)
      fprintf (fp, "%8.2e ", wallmax);
    else
      fprintf (fp, "%8.3f ", wallmax);
    fprintf (fp, "%5d ",gpustats[n].accum_max_warp);   // warp number for max
    
    wallmin = gpustats[n].accum_min / gpu_hz;          // min time for name across warps
    if (wallmin < 0.01)
      fprintf (fp, "%8.2e ", wallmin);
    else
      fprintf (fp, "%8.3f ", wallmin);	       
    fprintf (fp, "%5d ",gpustats[n].accum_min_warp);   // warp number for min
    
    count_max = gpustats[n].count_max;
    if (count_max < PRTHRESH)
      fprintf (fp, "%9lu ", count_max);                // max count for region "name"
    else
      fprintf (fp, "%9.1e ", (float) count_max);
    fprintf (fp, "%5d ",gpustats[n].count_max_warp);   // warp which accounted for max times
    
    count_min = gpustats[n].count_min;                
    if (count_min < PRTHRESH)
      fprintf (fp, "%9lu ", count_min);                // min count for region "name"
    else
      fprintf (fp, "%9.1e ", (float) count_min);
    fprintf (fp, "%5d ",gpustats[n].count_min_warp);   // warp which accounted for max times

    self = gpustats[n].count_max * self_ohdgpu[0] / gpu_hz;     // self ohd est
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
#pragma acc kernels copyout(ret, hashmem, regionmem)
  ret = GPTLget_memstats_gpu (hashmem, regionmem);
  fprintf (fp, "\n");
  fprintf (fp, "Total GPTL GPU memory usage = %g KB\n", (hashmem[0] + regionmem[0])*.001);
  fprintf (fp, "Components:\n");
  fprintf (fp, "Hashmem                     = %g KB\n" 
               "Regionmem                   = %g KB\n", hashmem[0]*.001, regionmem[0]*.001);
}
