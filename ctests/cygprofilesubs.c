#include <stdio.h>

extern void mult (int, float *);
extern void add (int, float *);
extern void multadd (int, float *);
extern void division (int, float *);

void callsubs (int niter)
{
  float zero = 0.;

  mult (niter, &zero);
  add (niter, &zero);
  multadd (niter, &zero);
  division (niter, &zero);
}

void mult (int niter, float *val)
{
  int i;

  printf ("Starting mult\n");
  for (i = 0; i < niter; i++) {
    *val *= i;
  }
}

void add (int niter, float *val)
{
  int i;

  printf ("Starting add\n");
  for (i = 0; i < niter; i++) {
    *val += i;
  }
}

void multadd (int niter, float *val)
{
  int i;

  printf ("Starting multadd\n");
  for (i = 0; i < niter; i++) {
    *val += i*0.1;
  }
}

void division (int niter, float *val)
{
  int i;

  printf ("Starting division\n");
  for (i = 1; i <= niter; i++) {
    *val /= i;
  }
}
