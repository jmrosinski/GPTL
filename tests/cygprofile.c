#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gptl.h"

extern void callsubs (int);

int main (int argc, char **argv)
{
  int niter = 1000;
  int ret;
  int nregions;
  
  if (argc == 2) {
    niter = atoi (argv[1]);
  } else if (argc > 2) {
    printf ("Usage: %s loop_length\n", argv[0]);
  }

  GPTLsetoption (GPTLabort_on_error, 0);

  if ((ret = GPTLinitialize ()) != 0) {
    printf ("%s: GPTLinitialize failure\n", argv[0]);
    return -1;
  }
  callsubs (niter);
  GPTLpr (0);

  printf ("%s: Testing GPTLget_nregions...\n", argv[0]);
  if ((ret = GPTLget_nregions (0, &nregions)) < 0) {
    printf ("%s: GPTLget_nregions failure\n", argv[0]);
    return -1;
  }

  if (nregions < 1) {
    printf ("%s: nregions should be at least 1 got %d\n", argv[0], nregions);
    return -1;
  }

  (void) GPTLfinalize ();
  return 0;
}
