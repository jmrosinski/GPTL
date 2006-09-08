#include <sys/time.h>  /* gettimeofday */
#include <sys/times.h> /* times */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>    /* gettimeofday */

#ifdef THREADED_OMP
#include <omp.h>
#endif

#ifdef HAVE_PAPI
#include <papi.h>
#endif

#include "../gptl.h"

static void parsub (int, int, float *);

int main (int argc, char **argv)
{
  int nompiter;
  int iter;
  int ninvoke;
  int papiopt;
  int gptlopt;
  int val;
  int i;
  float *sum;

  GPTLPAPIprinttable ();
  /*
  while (1) {
    printf ("Enter PAPI option to enable, non-negative number when done\n");
    scanf ("%d", &papiopt);
    if (papiopt >= 0)
      break;
    if (GPTLsetoption (papiopt, 1) < 0)
      printf ("gptlsetoption failure\n");
  }

  printf ("GPTLwall           = 1\n");
  printf ("GPTLcpu            = 2\n");
  printf ("GPTLabort_on_error = 3\n");
  printf ("GPTLoverhead       = 4\n");
  while (1) {
    printf ("Enter GPTL option and enable flag, negative numbers when done\n");
    scanf ("%d %d", &gptlopt, &val);
    if (gptlopt <= 0)
      break;
    if (GPTLsetoption (gptlopt, val) < 0)
      printf ("gptlsetoption failure\n");
  }
  printf ("Enter number of iterations for threaded loop:\n");
  scanf ("%d", &nompiter);
  printf ("Enter number of invocations of overhead portion:\n");
  scanf ("%d", &ninvoke);
  printf ("nompiter=%d ninvoke=%d\n", nompiter, ninvoke);

  */

  GPTLsetoption (PAPI_TOT_CYC, 1);
  GPTLsetoption (PAPI_FML_INS, 1);
  GPTLsetoption (GPTLoverhead, 0);

  nompiter = 4;
  ninvoke

  if ( ! (sum = malloc (ninvoke*sizeof (float))))
    exit (1);
  for (i = 0; i < ninvoke; ++i)
    sum[i] = 0.;

  GPTLinitialize ();
  GPTLstart ("total");

#pragma omp parallel for private (iter)

  for (iter = 0; iter < nompiter; ++iter) {
    parsub (iter, ninvoke, &sum[iter]);
  }

  GPTLstop ("total");
  GPTLpr (0);
  GPTLfinalize ();
}

void parsub (int iter,
	     int ninvoke,
	     float *sum)
{
  int i;
  int n;

  GPTLstart ("parsub");
  GPTLstart ("addition");
  for (n = 0; n < ninvoke; ++n) {
    for (i = 0; i < iter; ++i) {
      *sum += i + n;
    }
  }
  GPTLstop ("addition");

  GPTLstart ("FMA");
  for (n = 0; n < ninvoke; ++n) {
    for (i = 0; i < iter; ++i) {
      *sum += n*(i*1.1 + 7.);
    }
  }
  GPTLstop ("FMA");

  GPTLstart ("division");
  for (n = 0; n < ninvoke; ++n) {
    for (i = 0; i < iter; ++i) {
      *sum /= i + n;
    }
  }
  GPTLstop ("division");
  GPTLstop ("parsub");
}
