/*
** memusage.c
**
** Author: Jim Rosinski
**   Credit to Chuck Bardeen for MACOS section (__APPLE__ ifdef)
**
** Routines to gather and print stats about current memory usage
*/

#include "config.h"       // Must be first include
#include "gptl.h"         // function prototypes
#include "private.h"
#include "thread.h"
#include "memusage.h"
#include "util.h"

#include <sys/time.h>     // getrusage
#include <sys/resource.h> // getrusage
#include <unistd.h>       // sysconf

#ifdef __APPLE__
#include <stdint.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

static inline void set_fp_procsiz (void);

// namespace for variables and functions used elsewhere in GPTL
namespace memusage {
  float growth_pct = 0.;
  
  extern "C" void print_memstats (FILE *fp, Timer **timers, int tablesize)
  {
    Timer *ptr;               // walk through linked list
    float pchmem = 0.;        // parent/child array memory usage
    float regionmem = 0.;     // timer memory usage
    float papimem = 0.;       // PAPI stats memory usage
    float hashmem;            // hash table memory usage
    float callstackmem;       // callstack memory usage
    float totmem;             // total GPTL memory usage
    int numtimers;            // number of timers
    int t;
    
    hashmem = (float) sizeof (Hashentry) * tablesize * thread::max_threads;  // fixed size of table
    callstackmem = (float) sizeof (Timer *) * MAX_STACK * thread::max_threads;
    for (t = 0; t < thread::nthreads; t++) {
      numtimers = 0;
      for (ptr = timers[t]->next; ptr; ptr = ptr->next) {
	++numtimers;
	pchmem  += (float) sizeof (Timer *) * (ptr->nchildren + ptr->nparent);
      }
      hashmem   += (float) numtimers * sizeof (Timer *);
      regionmem += (float) numtimers * sizeof (Timer);
#ifdef HAVE_PAPI
      papimem += (float) numtimers * sizeof (Papistats);
#endif
    }
    
    totmem = hashmem + regionmem + pchmem + callstackmem;
    fprintf (fp, "\n");
    fprintf (fp, "Total GPTL memory usage = %g KB\n", totmem*.001);
    fprintf (fp, "Components:\n");
    fprintf (fp, "Hashmem                 = %g KB\n" 
	     "Regionmem               = %g KB (papimem portion = %g KB)\n"
	     "Parent/child arrays     = %g KB\n"
	     "Callstackmem            = %g KB\n",
	     hashmem*.001, regionmem*.001, papimem*.001, pchmem*.001, callstackmem*.001);
    
    thread::print_threadmapping (fp);
  }
}

// Start of user-callable functions
/*
** GPTLget_memusage: 
**
**   Returns current resident set size in MB
**
**   Return value: 0  = success
**                 -1 = failure
*/
extern "C" int GPTLget_memusage (float *rss_out)       // resident set size in MB
{
  static const float convert2mb = 1./1024.; // getrusage returns results in KB
  static const char *thisfunc = "GPTLget_memusage";

#ifdef HAVE_GETRUSAGE
  struct rusage usage;                      // structure filled in by getrusage

  if (getrusage (RUSAGE_SELF, &usage) < 0)
    return util::error ("%s: Failure from getrusage\n", thisfunc);
  
  *rss_out = (float) (usage.ru_maxrss * convert2mb);
#else
  return util::error ("%s: getrusage not available\n", thisfunc);
#endif
  return 0;
}

/*
** print_memusage:
**
**   Prints info about memory usage of this process by calling get_memusage.
**
**   Return value: 0  = success
**                 -1 = failure
*/
int GPTLprint_memusage (const char *str)
{
  float rss;       // resident set size (returned from getrusage
  static const char *thisfunc = "GPTLprint_memusage";
  
  if (GPTLget_memusage (&rss) < 0)
    return util::error ("%s: Failure from GPTLget_memusage", thisfunc);
  
  printf ("%s: %s rss=%f MB\n", thisfunc, str, rss);
  return 0;
  }

/*
** get_procsiz
**
**   Obtain process size and RSS for calling process
**
**   Return value: 0  = success
**                 -1 = failure
*/
int GPTLget_procsiz (float *procsiz_out, float *rss_out)
{
  int procsiz;
  int rss;
  static const char *thisfunc = "GPTLget_procsiz";
  
#ifdef HAVE_SLASHPROC
  int dum[5];                // placeholders for unused return arguments
  FILE *fd;                  // file descriptor for fopen
  static const char *file = "/proc/self/statm";
  int pagesize;
  int ret;
  static float convert2mb = 0.;
  
  if (convert2mb == 0. && (pagesize = sysconf (_SC_PAGESIZE)) > 0)
    convert2mb = pagesize / (1024.*1024.);
  
  if ((fd = fopen (file, "r")) < 0)
    return util::error ("%s: bad attempt to open %s\n", thisfunc, file);
  
  // Read the desired data from the /proc filesystem directly into the output
  // arguments, close the file and return.
  if ((ret = fscanf (fd, "%d %d %d %d %d %d %d",
		     &procsiz, &rss, &dum[0], &dum[1], &dum[2], &dum[3], &dum[4])) < 1) {
    (void) fclose (fd);
    return util::error ("%s: fscanf failure\n", thisfunc);
  }
  (void) fclose (fd);
  *procsiz_out = procsiz*convert2mb;
  *rss_out     = rss*convert2mb;
    
#elif (defined __APPLE__)
  
  FILE *fd;
  char cmd[60];  
  static const float convert2mb = 1./1024.;  // Apple reports in 1KB sizes
  int pid = (int) getpid ();
  
  sprintf (cmd, "ps -o vsz -o rss -p %d | grep -v RSS", pid);
  fd = popen (cmd, "r");
    
  if (fd) {
    fscanf (fd, "%d %d", &procsiz, &rss);
    (void) pclose (fd);
    *procsiz_out = procsiz*convert2mb;
    *rss_out     = rss*convert2mb;
  }
#else
  util::warn ("%s: Neither HAVE_SLASHPROC nor __APPLE__ are set so outputting -1\n", thisfunc);
  *procsiz_out = -1.;
  *rss_out     = -1.;
#endif
  return 0;
}
  
// End of user-callable functions  

extern "C" void check_memusage (const char *str, const char *funcnam)
{
  float rss;
  static float rssmax = 0;           // max rss of the process (init to zero). Require thread=0?
  static FILE *fp_procsiz = 0;       // process size file pointer: init to 0 to use stderr
  
  (void) GPTLget_memusage (&rss);
  // Notify user when rss has grown by more than some percentage (default 0%)
  if (rss > rssmax*(1.0 + 0.01*memusage::growth_pct)) {
    rssmax = rss;
    // Once MPI is initialized, change file pointer for process size to rank-specific file      
    set_fp_procsiz ();
    if (fp_procsiz) {
      fprintf (fp_procsiz, "%s %s RSS grew to %8.2f MB\n", str, funcnam, rss);
      fflush (fp_procsiz);  // Not clear when this file needs to be closed, so flush
      } else {
      fprintf (stderr, "%s %s RSS grew to %8.2f MB\n", str, funcnam, rss);
    }
  }
}

// Local functions

// set_fp_procsiz: Change file pointer from stderr to point to "procsiz.<rank>" once
// MPI has been initialized
static inline void set_fp_procsiz ()
{
#ifdef HAVE_LIBMPI
  int ret;
  int flag;
  static bool check_mpi_init = true; // whether to check if MPI has been init (init to true)
  char outfile[15];

  // Must only open the file once. Also more efficient to only make MPI lib inquiries once
  if (check_mpi_init) {
    ret = MPI_Initialized (&flag);
    if (flag) {
      int world_iam;
      check_mpi_init = false;
      ret = MPI_Comm_rank (MPI_COMM_WORLD, &world_iam);
      sprintf (outfile, "procsiz.%6.6d", world_iam);
      fp_procsiz = fopen (outfile, "w");
    }
  }
#endif
}
