/*
** memusage.c
**
** Author: Jim Rosinski
**   Credit to Chuck Bardeen for MACOS section (__APPLE__ ifdef)
**
** get_memusage: 
**
**   Designed to be called from Fortran, returns information about memory
**   usage in each of 5 input int* args.  On Linux read from the /proc
**   filesystem because getrusage() returns placebos (zeros).  Return -1 for
**   values which are unavailable or ambiguous on a particular architecture.
**
**   Return value: 0  = success
**                 -1 = failure
*/

#include <sys/resource.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* _AIX is automatically defined when using the AIX C compilers */
#ifdef _AIX
#include <sys/times.h>
#endif

#ifdef IRIX64
#include <sys/time.h>
#endif

#ifdef HAVE_SLASHPROC

#include <sys/time.h>
#include <sys/types.h>

#endif

#include "gptl.h"       /* function prototypes */
#include "private.h"

static int set_convert2mb (void);

static double convert2mb = 0.;  /* convert pages to MB (init to unset) */

int GPTLget_memusage (int *size_out,        /* process size in MB */
		      int *rss_out,         /* resident set size in MB */ 
		      int *share_out,       /* share segment size in MB */
		      int *text_out,        /* text segment size in MB */
		      int *datastack_out)   /* datastack segment size in MB */
{
  int size;  /* raw process size returned from system */
  int rss;   /* raw rss returned from system */
  int text;  /* raw text size returned from system */
  static const char *thisfunc = "GPTLget_memusage";

#ifdef HAVE_SLASHPROC
  FILE *fd;                       /* file descriptor for fopen */
  static char *file = "/proc/self/statm";
  int dum;                        /* placeholder for unused return arguments */
  int share;
  int datastack;
#elif (defined __APPLE__)
  FILE *fd;
  char cmd[60];  
  int pid = (int) getpid ();
#else
  struct rusage usage;         /* structure filled in by getrusage */
#endif

  /* Set factor to convert what the system returns to MB, otherwise give up */
  if (set_convert2mb () < 0)
    return GPTLerror ("%s: Cannot determine how to convert to MB", thisfunc);

#ifdef HAVE_SLASHPROC
  if ((fd = fopen (file, "r")) < 0)
    return GPTLerror ("%s: bad attempt to open %s\n", thisfunc, file);

  /*
  ** Read the desired data from the /proc filesystem directly into the output
  ** arguments, close the file and return.
  */
  (void) fscanf (fd, "%d %d %d %d %d %d %d", 
		 &size, &rss, &share, &text, &dum, &datastack, &dum);
  (void) fclose (fd);

  *size_out      = (int) (size      * convert2mb);
  *rss_out       = (int) (rss       * convert2mb);
  *share_out     = (int) (share     * convert2mb);
  *text_out      = (int) (text      * convert2mb);
  *datastack_out = (int) (datastack * convert2mb);

#elif (defined __APPLE__)

  sprintf (cmd, "ps -o vsz -o rss -o tsiz -p %d | grep -v RSS", pid);
  fd = popen (cmd, "r");

  if (fd) {
    fscanf (fd, "%d %d %d", &size, &rss, &text);
    *share_out     = -1;
    *datastack_out = -1;
    (void) pclose (fd);
    *size_out      = (int) (size * convert2mb);
    *rss_out       = (int) (rss  * convert2mb);
    *text_out      = (int) (text * convert2mb);
  }

#else

  struct rusage usage;         /* structure filled in by getrusage */

  if (getrusage (RUSAGE_SELF, &usage) < 0)
    return GPTLerror ("%s: Failure from getrusage", thisfunc);
  
  *size      = -1;
  *rss_out   = (int) (usage.ru_maxrss * convert2mb);
  *share     = -1;
  *text      = -1;
  *datastack = -1;
#ifdef IRIX64
  *datastack = (int) ((usage.ru_idrss + usage.ru_isrss) * convert2mb);
#endif

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
  int size;      /* process size (returned from OS) */
  int rss;       /* resident set size (returned from OS) */
  int share;     /* shared data segment size (returned from OS) */
  int text;      /* text segment size (returned from OS) */
  int datastack; /* data/stack size (returned from OS) */
  static const char *thisfunc = "GPTLprint_memusage";
  
  if (GPTLget_memusage (&size, &rss, &share, &text, &datastack) < 0)
    return GPTLerror ("%s: Failure from GPTLget_memusage", thisfunc);
  
  printf ("%s: %s size=%d MB rss=%d MB datastack=%d MB\n", 
	  thisfunc, str, size, rss, datastack);

  return 0;
}

/*
** set_pagesize:
**
**   Determine if possible the size of a page
*/
static int set_convert2mb ()
{
#if (defined HAVE_SLASHPROC)
  int pagesize;

  if (convert2mb == 0.) {
    if ((pagesize = sysconf (_SC_PAGESIZE)) > 0) {
      convert2mb = pagesize / (1024.*1024.);
    } else {
      return -1;
    }
  }

#elif (defined __APPLE__)

  if (convert2mb == 0.)
    convert2mb = 1./1024.;   // Apple reports in 1 KB sizes

#else

  if (convert2mb == 0.)
    convert2mb = 1./1024.;   // getrusage reports in 1 KB sizes

#endif
  return 0;
}
