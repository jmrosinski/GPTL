#include <stdio.h>
#include <stdlib.h>  /* malloc, exit */
#include <unistd.h>  /* getopt */
#include <string.h>  /* memset */
#include <papi.h>
#include "../gptl.h"

int main (int argc, char **argv)
{
  int looplen = 1000000;
  int c;
  int papiopt;
  double *a, *b;
  int *ia, *ib;

  extern void sub (int, double *, double *, int *, int *);
  extern char *optarg;

  while ((c = getopt (argc, argv, "p:")) != -1) {
    switch (c) {
    case 'p':
      if ((PAPI_event_name_to_code (optarg, &papiopt)) != 0) {
	printf ("Failure from PAPI_event_name_to_code\n");
	exit (1);
      }
      if (GPTLsetoption (papiopt, 1) < 0) {
	printf ("Failure from GPTLsetoption (%s,1)\n", optarg);
	exit (1);
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      exit (1);
    }
  }

  if (!(a = malloc (looplen * sizeof (a)))) {
    printf ("malloc failure\n");
    exit (1);
  }
  memset (a, 0, looplen*sizeof (a));

  if (!(b = malloc (looplen * sizeof (b)))) {
    printf ("malloc failure\n");
    exit (1);
  }
  memset (b, 0, looplen*sizeof (b));

  if (!(ia = malloc (looplen * sizeof (ia)))) {
    printf ("malloc failure\n");
    exit (1);
  }
  memset (ia, 0, looplen*sizeof (ia));

  if (!(ib = malloc (looplen * sizeof (ib)))) {
    printf ("malloc failure\n");
    exit (1);
  }
  memset (ib, 0, looplen*sizeof (ib));

  (void) GPTL_PAPIlibraryinit ();
  (void) GPTLsetoption (GPTL_CI, 1);
  (void) GPTLsetoption (GPTL_IPC, 1);

  if (GPTLinitialize () != 0) {
    printf ("Failure from GPTLinitialize\n");
    exit (1);
  }

  GPTLstart ("total");
  sub (looplen, a, b, ia, ib);
  GPTLstop ("total");

  GPTLpr (0);
  return 0;
}

void sub (int looplen, double *a, double *b, int *ia, int *ib)
{
  int i;

  GPTLstart ("Million_A=c*B");
  for (i = 0; i < looplen; ++i) {
    a[i] = 0.1*b[i];
  }
  GPTLstop ("Million_A=c*B");

  GPTLstart ("Million_IA=ic*IB");
  for (i = 0; i < looplen; ++i) {
    ia[i] = 3*ib[i];
  }
  GPTLstop ("Million_IA=ic*IB");
}
