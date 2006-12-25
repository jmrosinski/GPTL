#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include "../gptl.h"

float add (int);
float multiply (int);
float multadd (int);
float divide (int);

int main ()
{
  char cmd[256];
  int iter;
  int n;
  const int niter = 128;
  typedef struct {
    int counter;
    float (*funcptr)();
    char *name;
  } Counter;

  Counter tests[] = {
    {PAPI_FAD_INS, add,      "addition"},
    {PAPI_FML_INS, multiply, "multiplication"},
    {PAPI_FMA_INS, multadd,  "mult-add"},
    {PAPI_FDV_INS, divide,   "division"}
  };

  int numtests = sizeof (tests) / sizeof (Counter);

  for (n = 0; n < numtests; n++) {
    if (GPTLsetoption (tests[n].counter, 1) < 0) {
      printf ("Skipping test %s: not available\n", tests[n].name);
      continue;
    }
    if (GPTLsetoption (PAPI_TOT_CYC, 1) < 0)
      printf ("Total instructions count not available\n");

    if (GPTLinitialize () < 0) {
      if (GPTLfinalize () < 0)
	exit (2);
      
      continue;
    }
      
#pragma omp parallel for private (iter)
      
    for (iter = 1; iter <= niter; iter++) {
      (void) tests[n].funcptr (0);
    }

    if (GPTLpr (0) < 0)
      exit (3);

    if (GPTLfinalize () < 0)
      exit (4);

    sprintf (cmd, "mv timing.0 timing.0.%s", tests[n].name);
    (void) system (cmd);
    printf ("Output file timing.0.%s complete\n", tests[n].name);
  }
}

float add (int iter)
{
  int i;
  int looplen = iter * 1000000;
  float val = 0.;
  char string[16];

  sprintf (string, "add_%de6", iter);
  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val += i;

  if (GPTLstop (string) < 0)
    exit (1);
  
  return val;
}

float multiply (int iter)
{
  int i;
  int looplen = iter * 1000000;
  float val = 1./(looplen * looplen);
  char string[16];

  sprintf (string, "multiply_%de6", iter);
  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val *= i;

  if (GPTLstop (string) < 0)
    exit (1);
  
  return val;
}

float multadd (int iter)
{
  int i;
  int looplen = iter * 1000000;
  float val = 1./(looplen * looplen);
  char string[16];

  sprintf (string, "multadd_%de6", iter);
  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val *= i - looplen;

  if (GPTLstop (string) < 0)
    exit (1);
  
  return val;
}


float divide (int iter)
{
  int i;
  int looplen = iter * 1000000;
  float val = looplen * looplen;
  char string[16];

  sprintf (string, "divide_%de6", iter);
  if (GPTLstart (string) < 0)
    exit (1);

  for (i = 1; i <= looplen; ++i)
    val /= i;

  if (GPTLstop (string) < 0)
    exit (1);

  return val;
}
