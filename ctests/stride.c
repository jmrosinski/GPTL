#include "../gptl.h"
#include <stdio.h>

int main ()
{
#define ARRLEN 10000000

  void writesub (float *, int, int);
  void readsub (float *, int, int);

  int nstride = 64;
  int looplen = ARRLEN / nstride;

  int id;
  int n;

  /* Make arr static to keep off stack */

  static float arr[ARRLEN];

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

void writesub (float *arr, int n, int looplen)
{
  int i;
  char string[128];

  if (n > 8 && n % 2 != 0)
    GPTLdisable ();

  sprintf (string, "write_len%dstride%d", looplen, n);
  GPTLstart (string);
  for (i = 0; i < looplen; i++) {
    arr[n*i] = i;
  }
  GPTLstop (string);
  GPTLenable ();
}

void readsub (float *arr, int n, int looplen)
{
  int i;
  char string[128];
  float x = 0.;

  if (n > 8 && n % 2 != 0)
    GPTLdisable ();

  sprintf (string, "read_len%dstride%d", looplen, n);
  GPTLstart (string);
  for (i = 0; i < looplen; i++) {
    x += arr[n*i];
  }
  GPTLstop (string);
  GPTLenable ();
  arr[0] = x;
}
