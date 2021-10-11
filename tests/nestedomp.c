#include "config.h"
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include "gptl.h"

int main ()
{
  int m, n;            /* inner, outer nested loop indices */
  int t;               /* linear thread number */
  const int nsize = 3; // outer thread dimension
  const int msize = 2; // inner thread dimension
  double value;        /* return from GPTLget_wallclock */
  int ret;
  void sub (const int, const int, const int);
  
  omp_set_num_threads (nsize*msize);  // 3 outer x 2 inner threads
  omp_set_nested (1);

  ret = GPTLinitialize ();
#pragma omp parallel for private (n) num_threads(nsize)
  for (n = 0; n < nsize; ++n) {
    // Nested OMP in a test when GPTL built without it enabled results in a race condition
#ifdef ENABLE_NESTEDOMP
#pragma omp parallel for private (m, ret) num_threads(msize)
#endif
    for (m = 0; m < msize; ++m) {
      ret = GPTLstart ("sub");
      sub (m, n, msize);
      ret = GPTLstop ("sub");
    }
  }
  ret = GPTLpr (0);

  // Getting results for thread 1 should always succeed whether nesting enabled or not
  t = 1;
  ret = GPTLget_wallclock ("sub", t, &value);
  if (ret != 0) {
    printf ("Failure to get wallclock for t=%d\n", t);
    return 1;
  }

  // Test getting thread nsize*msize - 1. If nested omp enabled it should succeed, otherwise fail
  t = nsize*msize - 1;  // last thread
  ret = GPTLget_wallclock ("sub", t, &value);
#ifdef ENABLE_NESTEDOMP
  if (ret != 0) {
    printf ("Failure to get wallclock for t=%d\n", t);
    return 1;
  }
#else
  if (ret == 0) {
    printf ("Success getting wallclock for t=%d when it should fail\n", t);
    return 1;
  }
#endif
  return 0;
}

void sub (const int m, const int n, const int msize)
{
  int sleep_usecs = (useconds_t) (n*msize + m) * 1000;

  (void) usleep (sleep_usecs);
}
