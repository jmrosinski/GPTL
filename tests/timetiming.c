#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/times.h>
main()
{
  int n;
  struct tms buf;
#ifdef HAVE_GETRUSAGE
  struct rusage r_usage;
#endif

  t_initialize();
  for (n = 0; n < 100000; n++) {
#ifdef HAVE_GETRUSAGE
    t_start("getrusage");
    (void) getrusage (RUSAGE_SELF, &r_usage);
    t_stop("getrusage");
#endif

    t_start("times");
    (void) times (&buf);
    t_stop("times");
  }
  t_pr(0);
  exit(0);
}

