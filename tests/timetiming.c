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

  GPTinitialize();
  for (n = 0; n < 100000; n++) {
#ifdef HAVE_GETRUSAGE
    GPTstart("getrusage");
    (void) getrusage (RUSAGE_SELF, &r_usage);
    GPTstop("getrusage");
#endif

    GPTstart("times");
    (void) times (&buf);
    GPTstop("times");
  }
  GPTpr(0);
  exit(0);
}

