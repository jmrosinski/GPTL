/*
** memusage.c
**
** Author: Jim Rosinski
**   Credit to Chuck Bardeen for MACOS section (__APPLE__ ifdef)
**
** get_memusage: 
**
**   Returns current resident set size in MB
**
**   Return value: 0  = success
**                 -1 = failure
*/

#include "config.h" /* Must be first include. */
#include <sys/time.h>     // getrusage
#include <sys/resource.h> // getrusage
#include <unistd.h>       // sysconf

#include "gptl.h"       /* function prototypes */
#include "private.h"

#ifdef __cplusplus
extern "C" {
#endif

int GPTLget_memusage (float *rss_out)       // resident set size in MB
{
  static const float convert2mb = 1./1024.; // getrusage returns results in KB
  static const char *thisfunc = "GPTLget_memusage";

#ifdef HAVE_GETRUSAGE
  struct rusage usage;                      // structure filled in by getrusage

  if (getrusage (RUSAGE_SELF, &usage) < 0)
    return GPTLerror ("%s: Failure from getrusage\n", thisfunc);
  
  *rss_out = (float) (usage.ru_maxrss * convert2mb);
#else
  return GPTLerror ("%s: getrusage not available\n", thisfunc);
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
    return GPTLerror ("%s: Failure from GPTLget_memusage", thisfunc);
  
  printf ("%s: %s rss=%f MB\n", thisfunc, str, rss);
  return 0;
}

/*
** get_procsiz
**
**   Read from /proc to obtain process size and RSS for calling process
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
    return GPTLerror ("%s: bad attempt to open %s\n", thisfunc, file);

  /*
  ** Read the desired data from the /proc filesystem directly into the output
  ** arguments, close the file and return.
  */
  if ((ret = fscanf (fd, "%d %d %d %d %d %d %d",
		     &procsiz, &rss, &dum[0], &dum[1], &dum[2], &dum[3], &dum[4])) < 1) {
    (void) fclose (fd);
    return GPTLerror ("%s: fscanf failure\n", thisfunc);
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
  GPTLwarn ("%s: Neither HAVE_SLASHPROC nor __APPLE__ are set so outputting -1\n", thisfunc);
  *procsiz_out = -1.;
  *rss_out     = -1.;
#endif
  return 0;
}

#ifdef __cplusplus
}
#endif
