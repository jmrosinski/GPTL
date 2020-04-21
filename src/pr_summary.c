#include "config.h" // Must be first include.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>          // sqrt

#include "private.h"
#include "gptl.h"
#include "gptlmpi.h"
#include "thread.h"
#ifdef HAVE_PAPI
#include "gptl_papi.h"     // GPTLnevents, GPTLeventlist
#endif

// MPI summary stats
typedef struct {
  unsigned long totcalls;  // number of calls to the region across threads and tasks
#ifdef HAVE_PAPI
  double papimax[MAX_AUX]; // max counter value across threads, tasks
  double papimin[MAX_AUX]; // max counter value across threads, tasks
  int papimax_p[MAX_AUX];  // task producing papimax
  int papimax_t[MAX_AUX];  // thread producing papimax
  int papimin_p[MAX_AUX];  // task producing papimin
  int papimin_t[MAX_AUX];  // thread producing papimin
#endif
  unsigned int notstopped; // number of ranks+threads for whom the timer is ON
  unsigned int tottsk;     // number of tasks which invoked this region
  float wallmax;           // max time across threads, tasks
  float wallmin;           // min time across threads, tasks
  float mean;              // accumulated mean
  float m2;                // from Chan, et. al.
  int wallmax_p;           // task producing wallmax
  int wallmax_t;           // thread producing wallmax
  int wallmin_p;           // task producing wallmin
  int wallmin_t;           // thread producing wallmin
  char name[MAX_CHARS+1];  // timer name
} Global;

// Local prototypes
static void get_threadstats (int, char *, Timer **, Global *);
static Timer *getentry_slowway (Timer *, char *);

/* 
** GPTLpr_summary_file: Gather and print MPI summary stats across threads and tasks.
**                      The communication algorithm is O(log nranks) so it easily scales to 
**                      thousands of ranks. Added local memory usage is
**                      2*(number_of_regions)*sizeof(Global) on each rank.
**
** Input arguments:
**   comm:    communicator (e.g. MPI_COMM_WORLD)
**   outfile: name of file to be written
*/

int GPTLpr_summary_file (MPI_Comm comm, const char *outfile)
{
  int ret;             // return code
  int iam;             // my rank
  int nranks;          // number of ranks in communicator
  int nregions;        // number of regions aggregated across all tasks
  int nregions_p;      // number of regions for a single task
  int n, nn;           // region index
  int i;               // index
  Timer *ptr;          // linked list pointer
  Timer **timers;      // array of timers
  int incr;            // increment for tree sum
  int twoincr;         // 2*incr
  int dosend;          // logical indicating whether to send this iteration
  int dorecv;          // logical indicating whether to recv this iteration
  int sendto;          // rank to send to
  int p;               // rank to recv fm
  int mnl;             // max name length across all threads and tasks
  MPI_Status status;   // required by MPI_Recv
  int extraspace;      // for padding to length of longest name
  int multithread;     // flag indicates multithreaded or not for any task
  int multithread_p;   // recvd flag for other processor indicates multithreaded or not
  Global *global = 0;  // stats to be printed accumulated across tasks
  Global *global_p = 0;// stats to be printed for a single task
  Global *sptr;        // realloc intermediate
  float delta;         // from Chan, et. al.
  float sigma;         // st. dev.
  unsigned int tsksum; // part of Chan, et. al. equation
  static const int tag = 98789;                         // tag for MPI message
  static const int nbytes = sizeof (Global);            // number of bytes to be sent/recvd
  FILE *fp = 0;        // file handle to write to
#ifdef HAVE_PAPI
  int e;               // event index
#endif

  static const char *gptlversion = GPTL_VERSIONINFO;
  static const char *thisfunc = "GPTLpr_summary_file";  // this function

  if ( ! GPTLis_initialized ())
    return GPTLerror ("%s: GPTLinitialize() has not been called\n", thisfunc);

  if ((ret = MPI_Comm_rank (comm, &iam)) != MPI_SUCCESS)
    return GPTLerror ("%s: Bad return from MPI_Comm_rank=%d\n", thisfunc, ret);

  if ((ret = MPI_Comm_size (comm, &nranks)) != MPI_SUCCESS)
    return GPTLerror ("%s rank %d: Bad return from MPI_Comm_size=%d\n", thisfunc, iam, ret);

  // Examine only thread 0 regions that have not been renamed due to long name (only applies
  // to auto-profiled routines). The "longname" caveat is important because the naming truncation
  // algorithm may have named the SAME region differently for different ranks
  timers = GPTLget_timersaddr ();
  nregions = 0;
  for (ptr = timers[0]->next; ptr; ptr = ptr->next)
    if ( ! ptr->longname)
      ++nregions;

  if (nregions < 1)
    GPTLwarn ("%s rank %d: nregions = 0\n", thisfunc, iam);
  else
    global = (Global *) GPTLallocate (nregions * sizeof (Global), thisfunc);

  // Gather per-thread stats based on thread 0 list.
  // Also discover length of longest region name for formatting
  n = 0;
  mnl = 0;
  multithread = (GPTLnthreads > 1);

  for (ptr = timers[0]->next; ptr; ptr = ptr->next) {
    if ( ! ptr->longname) {
      get_threadstats (iam, ptr->name, timers, &global[n]);
      mnl = MAX (strlen (ptr->name), mnl);

      // Initialize for calculating mean, st. dev.
      global[n].mean   = global[n].wallmax;
      global[n].m2     = 0.;
      global[n].tottsk = 1;
      ++n;
    }
  }

  /*
  ** If all ranks participate in a region, could use MPI_Reduce to get mean and variance.
  ** But we can't assume that, so instead code the parallel algorithm by hand. 
  ** Log(ntask) algorithm to gather results to a single task is Jim Rosinski's concoction.
  ** One-pass algorithm for gathering mean and standard deviation comes from Chan et. al.
  ** (1979) described in: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  ** Discovered by googling for "one pass standard deviation" which found the Wikipedia
  ** page pointing to the Chan et. al. work. I'm not enough of a statistical whiz to
  ** be able to map the simple 3-line algorithm in the Wikipedia page (see "Parallel 
  ** algorithm") to anything in the Chan et. al. work, but it does work.
  */
  for (incr = 1; incr < nranks; incr = twoincr) {
    twoincr = 2*incr;
    sendto = iam - incr;
    p = iam + incr;      // could rename p as recvfm

    // The && part of the next 2 stmts prevents sending to or receiving from
    // outside communicator bounds when nranks is not a power of 2
    dorecv = ((iam + twoincr) % twoincr == 0) && (p < nranks);
    dosend = ((iam + incr) % twoincr == 0) && (sendto > -1);
    if (dosend) {
      if (dorecv)
        printf ("%s: WARNING: iam=%d: dosend and dorecv both true: possible hang?\n", thisfunc, iam);

      if ((ret = MPI_Send (&nregions, 1, MPI_INT, sendto, tag, comm)) != MPI_SUCCESS)
        return GPTLerror ("%s rank %d: Bad return from MPI_Send=%d\n", thisfunc, iam, ret);
      // if nregions=0, don't send other info because "global" wasn't even allocated
      // Same logic MUST also be applied on the receiving end
      if (nregions > 0) {
	if ((ret = MPI_Send (&multithread, 1, MPI_INT, sendto, tag, comm)) != MPI_SUCCESS)
	  return GPTLerror ("%s rank %d: Bad return from MPI_Send=%d\n", thisfunc, iam, ret);
	if ((ret = MPI_Send (global, nbytes*nregions, MPI_BYTE, sendto, tag, comm)) != MPI_SUCCESS)
	  return GPTLerror ("%s rank %d: Bad return from MPI_Send=%d\n", thisfunc, iam, ret);
      }
    }

    if (dorecv) {
      if (dosend)
        printf ("%s: WARNING: iam=%d: dosend and dorecv both true: possible hang?\n", thisfunc, iam);

      if ((ret = MPI_Recv (&nregions_p, 1, MPI_INT, p, tag, comm, &status)) != MPI_SUCCESS)
        return GPTLerror ("%s rank %d: Bad return from MPI_Recv=%d\n", thisfunc, iam, ret);
      if (nregions_p > 0) {
	if ((ret = MPI_Recv (&multithread_p, 1, MPI_INT, p, tag, comm, &status)) != MPI_SUCCESS)
	  return GPTLerror ("%s rank %d: Bad return from MPI_Recv=%d\n", thisfunc, iam, ret);
	if (multithread_p)
	  multithread = true;

	global_p = (Global *) GPTLallocate (nregions_p * sizeof (Global), thisfunc);
	ret = MPI_Recv (global_p, nbytes*nregions_p, MPI_BYTE, p, tag, comm, &status);
	if (ret != MPI_SUCCESS)
	  return GPTLerror ("%s rank %d: Bad return from MPI_Recv=%d\n", thisfunc, iam, ret);
      }
      
      // Merge stats for task p with our current stats. Note nregions_p and/or nregions may be 0
      for (n = 0; n < nregions_p; ++n) {
        for (nn = 0; nn < nregions; ++nn) {
          if (STRMATCH (global_p[n].name, global[nn].name)) {
            break;
          }
        }
	
        if (nn == nregions) {
	  // "received" region is for a region not on our rank. Allocate additional space and 
	  // copy the data that was just received. Note nregions (local to our rank) can be 0, 
	  // in which case nn will certainly be 0. Our rank still needs to participate, passing
	  // along the just received data.
          ++nregions;
          sptr = (Global *) realloc (global, nregions * sizeof (Global));
          if ( ! sptr)
            return GPTLerror ("%s: realloc error", thisfunc);
          global = sptr;
	  // IMPORTANT: structure copy only works because it contains NO pointers (only arrays)
          global[nn] = global_p[n];
          mnl = MAX (strlen (global[nn].name), mnl);

        } else {  // A matching name was just received: Adjust stats accordingly

	  // Won't print this entry if it was on for any rank or thread
	  global[nn].notstopped += global_p[n].notstopped;
	  global[nn].totcalls   += global_p[n].totcalls; // count is cumulative
          if (global_p[n].wallmax > global[nn].wallmax) {
            global[nn].wallmax   = global_p[n].wallmax;
            global[nn].wallmax_p = global_p[n].wallmax_p;
            global[nn].wallmax_t = global_p[n].wallmax_t;
          }
          if (global_p[n].wallmin < global[nn].wallmin) {
            global[nn].wallmin   = global_p[n].wallmin;
            global[nn].wallmin_p = global_p[n].wallmin_p;
            global[nn].wallmin_t = global_p[n].wallmin_t;
          }

          // Mean, variance calcs. Cast to float avoids possible integer overflow
          tsksum = global_p[n].tottsk + global[nn].tottsk;
          delta  = global_p[n].mean   - global[nn].mean;
          global[nn].mean += (delta * global_p[n].tottsk) / tsksum;
          global[nn].m2   += global_p[n].m2 + 
            delta * delta * ((float) global_p[n].tottsk * global[nn].tottsk) / tsksum;
          global[nn].tottsk = tsksum;

#ifdef HAVE_PAPI
          for (e = 0; e < GPTLnevents; ++e) {
            if (global_p[n].papimax[e] > global[nn].papimax[e]) {
              global[nn].papimax[e]   = global_p[n].papimax[e];
              global[nn].papimax_p[e] = global_p[n].papimax_p[e];
              global[nn].papimax_t[e] = global_p[n].papimax_t[e];
            }
            if (global_p[n].papimin[e] < global[nn].papimin[e]) {
              global[nn].papimin[e]   = global_p[n].papimin[e];
              global[nn].papimin_p[e] = global_p[n].papimin_p[e];
              global[nn].papimin_t[e] = global_p[n].papimin_t[e];
            }
          }
#endif
        }
      }
      if (global_p)
	free (global_p); // done with received data this iteration
    }                    // End of "if (dorecv) {" block
  }                      // End of "for (incr =..." loop

  // Rank 0 contains the final results. Print them
  if (iam == 0) {
    if ( ! (fp = fopen (outfile, "w"))) {
      fp = stderr;
      printf ("%s: WARNING: file=%s cannot be opened for writing. Using stderr instead\n",
	      thisfunc, outfile);
    }

    // Print version info from configure to output file
    fprintf (fp, "GPTL version info: %s\n", gptlversion);

    // Print a warning if error() was ever called
    if (GPTLnum_errors () > 0) {
      fprintf (fp, "WARNING: GPTLerror was called at least once during the run.\n");
      fprintf (fp, "Please examine your output for error messages beginning with GPTL...\n");
    }

    // Print heading
    fprintf (fp, "Total ranks in communicator=%d\n", nranks);
    fprintf (fp, "nthreads on rank 0=%d\n", GPTLnthreads);
    fprintf (fp, "'N' used for mean, std. dev. calcs.: 'ncalls'/'nthreads'\n");
    fprintf (fp, "'ncalls': number of times the region was invoked across tasks and threads.\n");
    fprintf (fp, "'nranks': number of ranks which invoked the region.\n");
    fprintf (fp, "mean, std. dev: computed using per-rank max time across all threads on each rank\n");
    fprintf (fp, "wallmax and wallmin: max, min time across tasks and threads.\n");

    fprintf (fp, "\nname");
    extraspace = mnl - strlen ("name");
    for (n = 0; n < extraspace; ++n)
      fprintf (fp, " ");
    fprintf (fp, "   ncalls nranks mean_time   std_dev   wallmax (rank  ");
    if (multithread)
      fprintf (fp, "thread");
    fprintf (fp, ")   wallmin (rank  ");
    if (multithread)
      fprintf (fp, "thread");
    fprintf (fp, ")");

#ifdef HAVE_PAPI
    for (e = 0; e < GPTLnevents; ++e) {
      fprintf (fp, " %8.8smax (rank  ", GPTLeventlist[e].event.str8);
      if (multithread)
        fprintf (fp, "thread");
      fprintf (fp, ")");

      fprintf (fp, " %8.8smin (rank  ", GPTLeventlist[e].event.str8);
      if (multithread)
        fprintf (fp, "thread");
      fprintf (fp, ")");
    }
#endif
    fprintf (fp, "\n");

    // Loop over regions and print summarized timing stats
    for (n = 0; n < nregions; ++n) {
      fprintf (fp, "%s", global[n].name);
      extraspace = mnl - strlen (global[n].name);

      for (i = 0; i < extraspace; ++i)
        fprintf (fp, " ");

      // Don't print stats if the timer is currently on for any thread or task: too dangerous 
      // since the timer needs to be stopped to have currently accurate timings
      if (global[n].notstopped > 0) {
	fprintf (fp, " NOT PRINTED: timer is currently ON for %d threads\n", 
		 global[n].notstopped);
	continue;
      }

      if (global[n].tottsk > 1)
        sigma = sqrt ((double) global[n].m2 / (global[n].tottsk - 1));
      else
        sigma = 0.;

      if (multithread) {  // Threads and tasks
        if (global[n].totcalls < PRTHRESH) {
          fprintf (fp, " %8lu %6u %9.3f %9.3f %9.3f (%6d %5d) %9.3f (%6d %5d)", 
                   global[n].totcalls, global[n].tottsk, global[n].mean, sigma, 
                   global[n].wallmax, global[n].wallmax_p, global[n].wallmax_t, 
                   global[n].wallmin, global[n].wallmin_p, global[n].wallmin_t);
        } else {
          fprintf (fp, " %8.1e %6u %9.3f %9.3f %9.3f (%6d %5d) %9.3f (%6d %5d)", 
                   (float) global[n].totcalls, global[n].tottsk, global[n].mean, sigma, 
                   global[n].wallmax, global[n].wallmax_p, global[n].wallmax_t, 
                   global[n].wallmin, global[n].wallmin_p, global[n].wallmin_t);
        }
      } else {  // No threads
        if (global[n].totcalls < PRTHRESH) {
          fprintf (fp, " %8lu %6u %9.3f %9.3f %9.3f (%6d) %9.3f (%6d)", 
                   global[n].totcalls, global[n].tottsk, global[n].mean, sigma, 
                   global[n].wallmax, global[n].wallmax_p, 
                   global[n].wallmin, global[n].wallmin_p);
        } else {
          fprintf (fp, " %8.1e %6u %9.3f %9.3f %9.3f (%6d) %9.3f (%6d)", 
                   (float) global[n].totcalls, global[n].tottsk, global[n].mean, sigma, 
                   global[n].wallmax, global[n].wallmax_p, 
                   global[n].wallmin, global[n].wallmin_p);
        }
      }

#ifdef HAVE_PAPI
      for (e = 0; e < GPTLnevents; ++e) {
        if (multithread)
          fprintf (fp, " %8.2e    (%6d %5d)", 
                   global[n].papimax[e], global[n].papimax_p[e], 
                   global[n].papimax_t[e]);
        else
          fprintf (fp, " %8.2e    (%6d)", 
                   global[n].papimax[e], global[n].papimax_p[e]);

        if (multithread)
          fprintf (fp, " %8.2e    (%6d %5d)", 
                   global[n].papimin[e], global[n].papimin_p[e], 
                   global[n].papimin_t[e]);
        else
          fprintf (fp, " %8.2e    (%6d)", 
                   global[n].papimin[e], global[n].papimin_p[e]);
      }
#endif
      fprintf (fp, "\n");
    }
    if (fp != stderr && fclose (fp) != 0)
      fprintf (stderr, "Attempt to close %s failed\n", outfile);
  }
  if (global)
    free (global);
  return 0;
}

// GPTLpr_summary: wrapper writes to "timing.summary"
int GPTLpr_summary (MPI_Comm comm)       /* communicator */
{
  static const char *outfile = "timing.summary";   // file to write to
  return GPTLpr_summary_file (comm, outfile);
}

/* 
** get_threadstats: gather stats for timer "name" over all threads
**
** Input arguments:
**   iam:    my rank
**   name:   timer name
**   timers: array of linked lists of timers
**   global: pointer to struct containing stats
** Output arguments:
**   global: max/min stats over all threads
*/
static void get_threadstats (int iam, char *name, Timer **timers, Global *global)
{
  int t;
  Timer *ptr;
  static const char *thisfunc = "get_threadstats";

  // This memset fortuitiously initializes the process values to master (0)
  memset (global, 0, sizeof (Global));
  strcpy (global->name, name);

  for (t = 0; t < GPTLnthreads; ++t) {
    if ((ptr = getentry_slowway (timers[t]->next, name))) {
      // Won't print this entry if it was on for any rank or thread
      if (ptr->onflg)
	++global->notstopped;

      global->totcalls += ptr->count;

      if (ptr->wall.accum > global->wallmax) {
        global->wallmax   = ptr->wall.accum;
        global->wallmax_p = iam;
        global->wallmax_t = t;
      }

      // global->wallmin = 0 for first thread
      if (ptr->wall.accum < global->wallmin || global->wallmin == 0.) {
        global->wallmin   = ptr->wall.accum;
        global->wallmin_p = iam;
        global->wallmin_t = t;
      }
#ifdef HAVE_PAPI
      int e;
      for (e = 0; e < GPTLnevents; ++e) {
        double value;
        if (GPTL_PAPIget_eventvalue (GPTLeventlist[e].event.namestr, &ptr->aux, &value) != 0) {
          fprintf (stderr, "GPTL: %s: Bad return from GPTL_PAPIget_eventvalue\n", thisfunc);
          return;
        }
        if (value > global->papimax[e]) {
          global->papimax[e]   = value;
          global->papimax_p[e] = iam;
          global->papimax_t[e] = t;
        }
        
	// First thread value in global is zero
        if (value < global->papimin[e] || global->papimin[e] == 0.) {
          global->papimin[e]   = value;
          global->papimin_p[e] = iam;
          global->papimin_t[e] = t;
        }
      }
#endif
    }
  }
}

// getentry_slowway: Find entry name in the table via linear search.
// Simpler this way since entries could be manual or auto-instrumented.
static Timer *getentry_slowway (Timer *timer, char *name)
{
  Timer *ptr = 0;

  for (; timer; timer = timer->next) {
    if (STRMATCH (name, timer->name)) {
      ptr = timer;
      break;
    }
  }
  return ptr;
}
