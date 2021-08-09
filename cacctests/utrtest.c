#include <stdio.h>
#include <stdlib.h>
#include "gptl.h"
#include "gptl_acc.h"

int main (int argc, char **argv)
{
  double sum;
  int ret;
  int handle1, handle2, handle3, handle4, handle5, handle6, handle7, handle8;
  int handle1_gpu, handle2_gpu, handle3_gpu, handle4_gpu, handle5_gpu, handle6_gpu,
    handle7_gpu, handle8_gpu, handle_total;
  int khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu;

  extern void sub (int, int, double, char *, int);
#pragma acc routine seq
  extern void sub_gpu (int, int, double *, int);
  
  printf ("Purpose: assess accuracy of GPTL overhead estimates\n");
  
  // Retrieve information about the GPU. Need only cores_per_gpu
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &smcount, &cores_per_sm, &cores_per_gpu);
  
  ret = GPTLinitialize ();

  // Set CPU handles
  ret = GPTLinit_handle ("1e7x1", &handle8);
  ret = GPTLinit_handle ("1e6x10", &handle7);
  ret = GPTLinit_handle ("1e5x100", &handle6);
  ret = GPTLinit_handle ("1e4x1000", &handle5);
  ret = GPTLinit_handle ("1000x1e4", &handle4);
  ret = GPTLinit_handle ("100x1e5", &handle3);
  ret = GPTLinit_handle ("10x1e6", &handle2);
  ret = GPTLinit_handle ("1x1e7", &handle1);

  // Set GPU handles
#pragma acc parallel private(ret) \
  copyout(handle8_gpu,handle7_gpu,handle6_gpu,handle5_gpu,handle4_gpu, \
	  handle3_gpu,handle2_gpu,handle1_gpu,handle_total)
  {
    ret = GPTLinit_handle_gpu ("total", &handle_total);
    ret = GPTLinit_handle_gpu ("1e7x1", &handle8_gpu);
    ret = GPTLinit_handle_gpu ("1e6x10", &handle7_gpu);
    ret = GPTLinit_handle_gpu ("1e5x100", &handle6_gpu);
    ret = GPTLinit_handle_gpu ("1e4x1000", &handle5_gpu);
    ret = GPTLinit_handle_gpu ("1000x1e4", &handle4_gpu);
    ret = GPTLinit_handle_gpu ("100x1e5", &handle3_gpu);
    ret = GPTLinit_handle_gpu ("10x1e6", &handle2_gpu);
    ret = GPTLinit_handle_gpu ("1x1e7", &handle1_gpu);
  }
  ret = GPTLcudadevsync ();

  sum = 0.;
  ret = GPTLstart ("total");
  ret = GPTLstart ("total_cpu");
  sub (10000000, 1, sum, "1e7x1", handle8);
  sub (1000000, 10, sum, "1e6x10", handle7);
  sub (100000, 100, sum, "1e5x100", handle6);
  sub (10000, 1000, sum, "1e4x1000", handle5);
  sub (1000, 10000, sum, "1000x1e4", handle4);
  sub (100, 100000, sum, "100x1e5", handle3);
  sub (10, 1000000, sum, "10x1e6", handle2);
  sub (1, 10000000, sum, "1x1e7", handle1);
  ret = GPTLstop ("total_cpu");

  sum = 0.;
  // GPU loop: hardwire for 5 SMs, 128 cores per SM
#pragma acc parallel private (ret) \
  copyin(handle1_gpu,handle2_gpu,handle3_gpu,handle4_gpu,handle5_gpu,    \
	 handle6_gpu,handle7_gpu,handle8_gpu,handle_total,cores_per_gpu) \
  copy(sum) reduction(+:sum)

#pragma acc loop gang worker vector
  for (int n = 0; n < cores_per_gpu; ++n) {
    ret = GPTLstart_gpu (handle_total);
    sub_gpu (10000000, 1, &sum, handle8_gpu);
    sub_gpu (1000000, 10, &sum, handle7_gpu);
    sub_gpu (100000, 100, &sum, handle6_gpu);
    sub_gpu (10000, 1000, &sum, handle5_gpu);
    sub_gpu (1000, 10000, &sum, handle4_gpu);
    sub_gpu (100, 100000, &sum, handle3_gpu);
    sub_gpu (10, 1000000, &sum, handle2_gpu);
    sub_gpu (1, 10000000, &sum, handle1_gpu);
    ret = GPTLstop_gpu (handle_total);
  }
  ret = GPTLcudadevsync ();
  ret = GPTLstop ("total");
    
  ret = GPTLpr (-1);  // negative number means write to stderr
  ret = GPTLcudadevsync ();
  printf ("Final value of sum=%g\n", sum);
  return 0;
}

void printusemsg_exit ()
{
  printf ("Usage: utrtest [-e]\n");
  printf ("where -e enables expensive sequencing\n");
  exit (1);
}

void sub (int outer, int inner, double sum, char *name, int handle)
{
  int i, j, ret;

  for (i = 0; i < outer; ++i) {
    ret = GPTLstart_handle (name, &handle);
    for (j = 0; j < inner; ++j) {
      sum += j;
    }
    ret = GPTLstop_handle (name, &handle);
  }
}

#pragma acc routine seq
void sub_gpu (int outer, int inner, double *sum, int handle)
{
  int i, j, ret;

  for (i = 0; i < outer; ++i) {
    ret = GPTLstart_gpu (handle);
    for (j = 0; j < inner; ++j) {
      *sum += j;
    }
    ret = GPTLstop_gpu (handle);
  }
}
