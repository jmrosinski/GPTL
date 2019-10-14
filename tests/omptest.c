#include "gptl.h"
#include <stdio.h>
#include <omp.h>

extern void sub (int);
extern void subsub (void);
extern void tree_bottom (void);

int main ()
{
  int ret;
  int iter;
  
  ret = GPTLinitialize ();
  ret = GPTLstart ("main");
  ret = GPTLstart ("omp_outer");
#pragma omp parallel for private (iter)
  for (iter = 0; iter < 100; ++iter) {
    sub (iter);
  }
  ret = GPTLstop ("omp_outer");
  ret = GPTLstop ("main");

  ret = GPTLpr (0);
}

void sub (int iter)
{
  int ret;
  int mythread = omp_get_thread_num();

  ret = GPTLstart ("sub");
  printf ("iter=%d being processed by thread=%d\n", iter, mythread);
  subsub ();
  ret = GPTLstop ("sub");
}

void subsub (void)
{
  int ret;
  
  ret = GPTLstart ("subsub");
  tree_bottom ();
  ret = GPTLstop ("subsub");
}

void tree_bottom (void)
{
  int ret;
  
  ret = GPTLstart ("tree_bottom");
  ret = GPTLstop ("tree_bottom");
}
  
