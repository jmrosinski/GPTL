#include <stdio.h>
#include "../gptl.h"

int do_throw (void);
int empty_sub (void);

int main ()
{
  int i;
  int niter;
  int ret;

  if ((ret = GPTLinitialize ()) != 0) {
    printf ("exception: GPTLinitialize failure\n");
    return -1;
  }

  GPTLstart ("total");

  printf ("Enter number of iterations\n");
  scanf  ("%d", &niter);

  for (i = 0; i < niter; ++i) {
    GPTLstart ("do_nothing");
    GPTLstop  ("do_nothing");
    GPTLstart ("empty_sub");
    ret = empty_sub ();
    GPTLstop  ("empty_sub");
    GPTLstart ("exception");
    try {
      do_throw ();
    }
    catch (...) {
      GPTLstop ("exception");
    }
  }
  GPTLstop ("total");
  GPTLpr (0);
}

int empty_sub (void)
{
  static int i = 7;
  return i;
}

int do_throw (void)
{
  static int i = 7;
  throw (i);
  return i;
}
