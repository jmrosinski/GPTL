/** @file GPTL header file to be included in user code.
 *
 * @author Jim Rosinski
 */

#ifndef GPTL_H
#define GPTL_H

/*
** Options settable by a call to GPTLsetoption() (default in parens)
** These numbers need to be small integers because GPTLsetoption can
** be passed PAPI counters, and we need to avoid collisions in that
** integer space. PAPI presets are big negative integers, and PAPI
** native events are big positive integers.
*/
typedef enum {
  GPTLsync_mpi        = 0,  /* Synchronize before certain MPI calls (PMPI-mode only) */
  GPTLwall            = 1,  /* Collect wallclock stats (true) */
  GPTLcpu             = 2,  /* Collect CPU stats (false)*/
  GPTLabort_on_error  = 3,  /* Abort on failure (false) */
  GPTLoverhead        = 4,  /* Estimate overhead of underlying timing routine (true) */
  GPTLdepthlimit      = 5,  /* Only print timers this depth or less in the tree (inf) */
  GPTLverbose         = 6,  /* Verbose output (false) */
  GPTLpercent         = 9,  /* Add a column for percent of first timer (false) */
  GPTLpersec          = 10, /* Add a PAPI column that prints "per second" stats (true) */
  GPTLmultiplex       = 11, /* Allow PAPI multiplexing (false) */
  GPTLdopr_preamble   = 12, /* Print preamble info (true) */
  GPTLdopr_threadsort = 13, /* Print sorted thread stats (true) */
  GPTLdopr_multparent = 14, /* Print multiple parent info (true) */
  GPTLdopr_collision  = 15, /* Print hastable collision info (true) */
  GPTLdopr_memusage   = 27, /* Call GPTLprint_memusage when auto-instrumented */
  GPTLprint_method    = 16, /* Tree print method: first parent, last parent
			       most frequent, or full tree (most frequent) */
  GPTLtablesize       = 50, /* per-thread size of hash table */
  GPTLmaxthreads      = 51, /* maximum number of threads */
  GPTLonlyprint_rank0 = 52, // Restrict printout to rank 0 when MPI enabled

  // These are derived counters based on PAPI counters. All default to false
  GPTL_IPC           = 17, /* Instructions per cycle */
  GPTL_LSTPI         = 21, /* Load-store instruction fraction */
  GPTL_DCMRT         = 22, /* L1 miss rate (fraction) */
  GPTL_LSTPDCM       = 23, /* Load-stores per L1 miss */
  GPTL_L2MRT         = 24, /* L2 miss rate (fraction) */
  GPTL_LSTPL2M       = 25, /* Load-stores per L2 miss */
  GPTL_L3MRT         = 26  /* L3 read miss rate (fraction) */
} GPTL_Option;

/*
** Underlying wallclock timer: optimize for best granularity with least overhead.
** These numbers need not be distinct from the above because these are passed
** to GPTLsetutr() and the above are passed to GPTLsetoption()
*/
typedef enum {
  GPTLgettimeofday   = 1, /* ubiquitous but slow */
  GPTLnanotime       = 2, /* only available on x86 */
  GPTLmpiwtime       = 4, /* MPI_Wtime */
  GPTLclockgettime   = 5, /* clock_gettime */
  GPTLplacebo        = 7, /* do-nothing */
  GPTLread_real_time = 3  /* AIX only */
} GPTL_Funcoption;

typedef enum {
  GPTLfirst_parent  = 1,  /* first parent found */
  GPTLlast_parent   = 2,  /* last parent found */
  GPTLmost_frequent = 3,  /* most frequent parent (default) */
  GPTLfull_tree     = 4   /* complete call tree */
} GPTL_Method;

// All User-callable function prototypes except for MPI (see gptlmpi.h)
// They require C linkage
#ifdef __cplusplus
extern "C" {
#endif
  // In once.cc:
  int GPTLsetoption (const int, const int);
  int GPTLsetutr (const int option);
  int GPTLinitialize (void);
  int GPTLfinalize (void);

  // In gptl.cc:
  int GPTLinit_handle (const char *, int *);
  int GPTLstart (const char *);
  int GPTLstart_handle (const char *, int *);
  int GPTLstop (const char *);
  int GPTLstop_handle (const char *, int *);
  int GPTLstartstop_val (const char *, double);

  // In getter.cc:
  int GPTLstamp (double *, double *, double *);
  int GPTLquery (const char *, int, int *, int *, double *, double *, double *, long long *,
		 const int);
  int GPTLget_wallclock (const char *, int, double *);
  int GPTLget_wallclock_latest (const char *, int, double *);
  int GPTLget_nregions (int, int *);
  int GPTLget_regionname (int, int, char *, int);
  int GPTLget_threadwork (const char *, double *, double *);
  int GPTLget_eventvalue (const char *, const char *, int, double *);
  int GPTLget_count (const char *, int, int *);
  int GPTLnum_errors (void);
  int GPTLnum_warn (void);

  // In postprocess.cc:
  int GPTLpr (const int);
  int GPTLpr_file (const char *);
  
  // In setter.cc:
  int GPTLsetutr (const int);
  int GPTLreset (void);
  int GPTLreset_errors (void);
  int GPTLreset_timer (char *);
  int GPTLdisable (void);
  int GPTLenable (void);

  // In memusage.cc:
  int GPTLget_procsiz (float *, float *);
  int GPTLprint_memusage (const char *);
  int GPTLprint_rusage (const char *);
  int GPTLget_memusage (float *);

  // In gptl_papi.cc:
  int GPTLevent_name_to_code (const char *, int *);
  int GPTLevent_code_to_name (const int, char *);
#ifdef __cplusplus
}
#endif
#endif
