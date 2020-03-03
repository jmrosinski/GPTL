#include "gptl.h"
#include <stdio.h>
#include <omp.h>

extern void sub (int);

int main ()
{
  int ret;
  int iter;
  double value;
  static const char *thisprog = "omptest";

  omp_set_num_threads (2);
  ret = GPTLinitialize ();
  ret = GPTLstart ("main");
  ret = GPTLstart ("omp_loop");
#pragma omp parallel for private (iter)
  for (iter = 0; iter < 2; ++iter) {
    sub (iter);
  }
  ret = GPTLstop ("omp_loop");
  ret = GPTLstop ("main");

  // This test should succeed
  ret = GPTLget_wallclock ("sub", 1, &value);
  if (ret != 0) {
    printf ("%s: GPTLget_wallclock failure for thread 1\n", thisprog);
    return -1;
  }

  // This test should fail
  ret = GPTLget_wallclock ("sub", 2, &value);
  if (ret == 0) {
    printf ("%s: GPTLget_wallclock should have failed for thread 2\n", thisprog);
    return -1;
  }
  return 0;
}

void sub (int iter)
{
  int ret;
  int mythread = omp_get_thread_num();

  ret = GPTLstart ("sub");
  printf ("iter=%d being processed by thread=%d\n", iter, mythread);
  ret = GPTLstop ("sub");
}
