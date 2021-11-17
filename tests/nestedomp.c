#include "config.h"
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include "gptl.h"

int main ()
{
  int i, j, k;         // loop indices
  const int ksize = 4; // outer thread dimension
  const int jsize = 2; // inner thread dimension
  const int isize = 1; // inner loop dimension
  int outerworks = 0;  // outer threading works (stays false if outer threading disabled)
  int innerworks = 0;  // inner threading works (stays false if inner threading disabled)
  int ret;

  void sub (const int, const int, const int, const int, const int);

  // NOTE: omp_set_nested() is deprecated in favor of omp_set_max_active_levels()
  // BUT this approach fails under gcc 10.3.0. It works under Intel 2021.3.0
  omp_set_max_active_levels (2); // enable 2 levels of nesting
  ret = GPTLsetoption (GPTLmaxthreads, ksize + ksize*(jsize-1));
  ret = GPTLinitialize ();
  ret = GPTLstart ("total");

  // Uncomment the next line to run vtune
  //  for (int iter=0; iter<10000; ++iter){
#pragma omp parallel for private (k, ret) num_threads(ksize)
  for (k = 0; k < ksize; ++k) {
    // Nested OMP in a test when GPTL built without it enabled results in a race condition
    if (omp_get_thread_num () == ksize-1)
      outerworks = 1;
    ret = GPTLstart ("k_loop");
#pragma omp parallel for private (i, j, ret) num_threads(jsize)
    for (j = 0; j < jsize; ++j) {
      if (omp_get_thread_num () == jsize-1)
	innerworks = 1;
      ret = GPTLstart ("j_loop");
      for (i = 0; i < isize; ++i) {
	ret = GPTLstart ("sub");
	sub (i, j, k, isize, jsize);
	ret = GPTLstop ("sub");
      }
      ret = GPTLstop ("j_loop");
    }
    ret = GPTLstop ("k_loop");
  }
  // Uncomment the next line to run vtune
  //  }
  ret = GPTLstop ("total");
  ret = GPTLpr (0);
  if (outerworks && innerworks) {
    printf ("Nesting OMP to 2 levels works\n");
    return 0;
  } else {
    printf ("Nesting OMP to 2 levels fails: rerun configure with --disable-nestedomp\n");
    return -1;
  }
}

void sub (const int i, const int j, const int k, const int isize, const int jsize)
{
  int sleep_usecs = (useconds_t) (k*100 + j*10 + i)*1000;
  (void) usleep (sleep_usecs);
}
