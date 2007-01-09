#include <stdio.h>
#include <stdlib.h>

#include "../gptl.h"

int main ()
{
#define ARRLEN 10000000

  void writesub (double *, int, int);
  void readsub (double *, int, int);

  int nstride = 64;
  int looplen = ARRLEN / nstride;

  int id;
  int n;

  /* Make arr static to keep off stack */

  static double arr[ARRLEN];

  printf ("Test strided reads/writes to memory\n");

  GPTLsetoption (GPTLabort_on_error, 1);

#ifdef HAVE_PAPI
  if ((id = GPTL_PAPIname2id ("PAPI_TOT_CYC")) > 0)
    exit (id);
  if (GPTLsetoption (id, 1) < 0)
    exit (2);
#endif

  GPTLinitialize ();
  GPTLstart ("total");

  for (n = 1; n <= nstride; n++) {
    writesub (arr, n, looplen);
    readsub (arr, n, looplen);
  }

  GPTLstop ("total");
  GPTLpr (0);

  return 0;
}

void writesub (double *arr, int n, int looplen)
{
  int i;
  char string[128];

  if (n > 8 && n % 2 != 0)
    GPTLdisable ();

  sprintf (string, "write_len%dstride%d", looplen, n);
  GPTLstart (string);
  for (i = 0; i < looplen; i++) {
    arr[n*i] = 0.1*(n*i);
  }
  GPTLstop (string);
  GPTLenable ();
}

void readsub (double *arr, int n, int looplen)
{
  int i;
  char string[128];
  double x = 0.;
  double ref = 0.;
  double relerr;

  if (n > 8 && n % 2 != 0)
    GPTLdisable ();

  sprintf (string, "read_len%dstride%d", looplen, n);
  GPTLstart (string);
  for (i = 0; i < looplen; i++) {
    x += arr[n*i];
  }
  GPTLstop (string);
  GPTLenable ();

  for (i = 0; i < looplen; i++) {
    ref += 0.1*(n*i);
  }

  relerr = (ref - x) / (0.5*(ref + x));
  printf ("Relative error stride=%d niter=%d is %g percent\n",
	  n, looplen, relerr*100.);
}
