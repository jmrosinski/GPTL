#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papiStdEventDefs.h>
#endif

extern void callsubs (int);

int main (int argc, char **argv)
{
  int niter = 1000;  /* default number of */
  
  printf ("Purpose: test library with automatic per-function timing enabled\n");
  if (argc == 2) {
    niter = atoi (argv[1]);
  } else if (argc > 2) {
    printf ("Usage: %s loop_length\n");
  }
  printf ("Using %d iterations\n", niter);

  GPTLsetoption (GPTLabort_on_error, 1);

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_TOT_CYC, 1);
  GPTLsetoption (PAPI_FP_INS, 1);
#endif

  GPTLinitialize ();
  callsubs (niter);
  GPTLpr (0);

  (void) GPTLfinalize ();
  return 0;
}
