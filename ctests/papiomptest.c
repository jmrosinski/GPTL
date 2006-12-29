#include <stdio.h>
#include <stdlib.h>  /* atoi */
#include <unistd.h>  /* getopt */
#include <papi.h>
#include "../gptl.h"

float add (int);
float multiply (int);
float multadd (int);
float divide (int);

int main (int argc, char **argv)
{
  int iter;
  int n;
  int nompiter = 128;
  int looplen = 1000000;
  int papiopt;
  float ret;
  float zero = 0.;

  extern char *optarg;

  int c;

  printf ("Purpose: test PAPI and (optionally) OpenMP\n");

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
	  if ((papiopt = GPTL_PAPIname2str (optarg)) < 0) {
	    printf ("Failure from GPTL_PAPIname2str\n");
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

  if (GPTLinitialize () < 0) 
    exit (3);
	 
#pragma omp parallel for private (iter, zero)
      
    for (iter = 1; iter <= nompiter; iter++) {
      ret = add (looplen, &zero);
      ret = multiply (looplen, &zero);
      ret = multadd (looplen, &zero);
      ret = divide (looplen, &zero);
    }

    if (GPTLpr (0) < 0)
      exit (4);

    if (GPTLfinalize () < 0)
      exit (5);
  }
}

float add (int looplen, float *zero)
{
  int i;
  float val = *zero;
  char string[128];

  if (looplen < 1000000)
    sprintf (string, "%dadditions", looplen);
  else
    sprintf (string, "%10.3gadditions", looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val += i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

float multiply (int looplen, float *zero)
{
  int i;
  float val = *zero;
  char string[128];

  if (looplen < 1000000)
    sprintf (string, "%dmultiplies", looplen);
  else
    sprintf (string, "%10.3gmultiplies", looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val *= i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

float multadd (int looplen, float *zero)
{
  int i;
  float val = *zero;
  char string[128];

  if (looplen < 1000000)
    sprintf (string, "%dmultadds", looplen);
  else
    sprintf (string, "%10.3gmultadds", looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val += i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}

float divide (int looplen, float *zero)
{
  int i;
  float val = *zero;
  char string[128];

  if (looplen < 1000000)
    sprintf (string, "%ddivides", looplen);
  else
    sprintf (string, "%10.3gdivides", looplen);

  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val /= i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}
