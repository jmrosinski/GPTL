#include <stdio.h>
#include <unistd.h>
#ifdef THREADED_OMP
#include <omp.h>
#endif
#include "../gptl.h"

int main ()
{
  int m, n;            /* inner, outer nested loop indices */
  int t;               /* linear thread number */
  const int msize = 2; /* dimension M */
  double value;        /* return from GPTLget_wallclock */
  int ret;
  void sub (const int, const int, const int);
  
#ifdef THREADED_OMP
  omp_set_num_threads (6);  /* 3 outer x 2 inner threads */
  omp_set_nested (1);
#endif
  ret = GPTLinitialize ();
#pragma omp parallel for private (n) num_threads(3)
  for (n = 0; n < 3; ++n) {
#pragma omp parallel for private (m, ret) num_threads(2)
    for (m = 0; m < msize; ++m) {
      ret = GPTLstart ("sub");
      sub (m, n, msize);
      ret = GPTLstop ("sub");
    }
  }
#ifdef THREADED_OMP
  for (n = 0; n < 3; ++n) {
    for (m = 0; m < msize; ++m) {
      t = n*msize + m;
      ret = GPTLget_wallclock ("sub", t, &value);
      if (ret != 0) {
	printf ("Failure to get wallclock for t=%d\n", t);
	return 1;
      }
    }
  }
#endif
  ret = GPTLpr (0);
  return 0;
}

void sub (const int m, const int n, const int msize)
{
  int sleep_usecs = (useconds_t) (n*msize + m) * 1000;

  (void) usleep (sleep_usecs);
}
