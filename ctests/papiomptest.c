#include <stdio.h>
#include <stdlib.h>  /* atoi */
#include <unistd.h>  /* getopt */
#include <string.h>  /* memset */

#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

double add (int, int);
double multiply (int, int, double);
double multadd (int, int);
double divide (int, int);
double compare (int, int);

int main (int argc, char **argv)
{
  int nompiter = 128;
  int looplen = 1000000;
  int iter;
  int papiopt;

  extern char *optarg;

  int c;

  printf ("Purpose: test known-length loops with various floating point ops\n");
  printf ("Include PAPI and OpenMP, respectively, if enabled\n");
  printf ("Usage: %s [-l looplen] [-n nompiter] [-p papi_option_name]\n", argv[0]);

  while ((c = getopt (argc, argv, "l:n:p:")) != -1) {
    switch (c) {
	case 'l':
	  looplen = atoi (optarg);
	  printf ("Set looplen=%d\n", looplen);
	  break;
	case 'n':
	  nompiter = atoi (optarg);
	  printf ("Set nompiter=%d\n", nompiter);
	  break;
	case 'p':
	  if ((papiopt = GPTL_PAPIname2id (optarg)) >= 0) {
	    printf ("Failure from GPTL_PAPIname2id\n");
	    exit (1);
	  }
	  if (GPTLsetoption (papiopt, 1) < 0) {
	    printf ("Failure from GPTLsetoption (%s,1)\n", optarg);
	    exit (1);
	  }
	  break;
	default:
	  printf ("unknown option %c\n", c);
	  exit (2);
    }
  }
  
  printf ("Outer loop length (OMP)=%d\n", nompiter);
  printf ("Inner loop length=%d\n", looplen);

  if (GPTLsetoption (GPTLabort_on_error, 1) < 0)
    exit (4);

  GPTLinitialize ();
  GPTLstart ("total");
	 
#pragma omp parallel for private (iter)
      
  for (iter = 1; iter <= nompiter; iter++) {
    add (looplen, iter);
    multiply (looplen, iter, 0.);
    multadd (looplen, iter);
    divide (looplen, iter);
    compare (looplen, iter);
  }

  GPTLstop ("total");
  GPTLpr (0);
  if (GPTLfinalize () < 0)
    exit (6);

  return 0;
}

double add (int looplen, int iter)
{
  int i;
  char string[128];
  double val = iter;

  if (looplen < 1000000)
    sprintf (string, "%dadditions", looplen);
  else
    sprintf (string, "%-.3gadditions", (double) looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val += (double) i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

double multiply (int looplen, int iter, double zero)
{
  int i;
  char string[128];
  double val = iter;

  if (looplen < 1000000)
    sprintf (string, "%dmultiplies", looplen);
  else
    sprintf (string, "%-.3gmultiplies", (double) looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val *= zero;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

double multadd (int looplen, int iter)
{
  int i;
  char string[128];
  double val = iter;

  if (looplen < 1000000)
    sprintf (string, "%dmultadds", looplen);
  else
    sprintf (string, "%-.3gmultadds", (double) looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val += ((double) i) * 0.00001;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

double divide (int looplen, int iter)
{
  int i;
  char string[128];
  double val = iter;

  if (looplen < 1000000)
    sprintf (string, "%ddivides", looplen);
  else
    sprintf (string, "%-.3gdivides", (double) looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val /= (double) i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

double compare (int looplen, int iter)
{
  int i;
  char string[128];
  double val = iter;

  if (looplen < 1000000)
    sprintf (string, "%ddivides", looplen);
  else
    sprintf (string, "%-.3gcompares", (double) looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 0; i < looplen; ++i)
    if (val < i)
      val = i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}
