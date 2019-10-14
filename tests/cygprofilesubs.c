#include "config.h"
#include <stdio.h>

extern void A (void);
extern void B (void);
extern void twice (void);

void callsubs (int niter)
{
  A();
}

void A (void)
{
  int n;
  
#pragma omp parallel for private(n)
  for (n = 0; n < 2; ++n) {
    B ();
  }
}

void B (void)
{
  twice ();
  twice ();
}

void twice (void)
{
}
