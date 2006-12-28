#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

extern void callsubs (int);

int main (int argc, char **argv)
{
  int niter = 1000;  /* default */
  
  printf ("Purpose: test library with automatic per-function timing enabled\n");
  if (argc == 2) {
    niter = atoi (argv[0]);
  } else if (argc > 2) {
    printf ("Usage: %s loop_length\n");
  }
  printf ("Using %d iterations\n", niter);

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_TOT_CYC, 1);
#endif

  GPTLinitialize ();
  callsubs (niter);
  GPTLpr (0);

  (void) GPTLfinalize ();
}
