#include <stdio.h>
#include "../gptl.h"
#include "../cuda/gptl.h"

int main ()
{
  int ret;
  int n;

#pragma acc routine (sub) seq

  printf ("simple: calling GPTLinitialize 1\n");
  ret = GPTLinitialize ();
  printf ("simple: calling GPTLinitialize 2\n");
  ret = GPTLinitialize ();

  ret = GPTLstart ("total");
  printf ("Entering kernels loop\n");

#pragma acc kernels copyout(ret)
  for (n=1; n<2; ++n) {
    ret = sub ();
  }

  printf ("Exiting kernels loop\n");
  ret = GPTLstop ("total");
  printf ("Calling GPTLpr\n");
  ret = GPTLpr (0);
  return ret;
}

#pragma acc routine seq
int sub ()
{
  int ret;
  char *v1;
  char *v2;

#pragma acc routine (GPTLstart_gpu) seq
#pragma acc routine (GPTLstop_gpu) seq
  ret = GPTLstart_gpu (v1);
  ret = GPTLstart_gpu (v2);
  ret = GPTLstop_gpu (v2);
  ret = GPTLstop_gpu (v1);
  return ret;
}
