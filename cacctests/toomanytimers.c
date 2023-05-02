#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <openacc.h>
#include "gptl.h"
#include "gptl_acc.h"

// Purpose: Verify correct behavior of setting handles and calling start
int main (int argc, char **argv)
{
  int ret;                 // return code
  int first_gpu_handle;
  int second_gpu_handle;
  int third_gpu_handle;
  int handle_success;      // number of successful calls to GPTLinit_handle. Should be 1
  int start_success;       // number of successful calls to GPTLstart_gpu. Should be 1

#ifndef ENABLE_GPUCHECKS
  printf ("toomanytimers: placebo since ENABLE_GPUCHECKS at build time was not defined\n");
  return 0;
#endif
  
  // Allow only 1 user timer (dimension of 2 due to GPTL_ROOT)
  ret = GPTLsetoption (GPTLmaxtimers_gpu, 2);

  // Initialize the GPTL library on CPU and GPU
  ret = GPTLinitialize ();

  // Define too many GPU timers
  handle_success = 0;
#pragma acc parallel private(ret) copy(handle_success) \
                     copyout(first_gpu_handle,second_gpu_handle,third_gpu_handle)
  {
    if ((ret = GPTLinit_handle_gpu ("first_gpu_timer", &first_gpu_handle)) == 0) {
      handle_success = 1;
      if ((ret = GPTLinit_handle_gpu ("second_gpu_timer", &second_gpu_handle)) == 0) {
	handle_success = 2;
	if ((ret = GPTLinit_handle_gpu ("third_gpu_timer", &third_gpu_handle)) == 0) {
	  handle_success = 3;
	}
      }
    }
  }
  ret = GPTLcudadevsync ();
  printf ("GPTLmaxtimers_gpu=2, GPTLinit_handle succeeded through %d user handles\n", handle_success);

  start_success = 0;
#pragma acc parallel private(ret) copy(start_success) \
                     copyin(first_gpu_handle,second_gpu_handle,third_gpu_handle)
  if ((ret = GPTLstart_gpu (first_gpu_handle)) == 0) {
    start_success = 1;
    if ((ret = GPTLstart_gpu (second_gpu_handle)) == 0) {
      start_success = 2;
      if ((ret = GPTLstart_gpu (third_gpu_handle)) == 0) {
	start_success = 3;
      }
    }
  }
  printf ("GPTLmaxtimers_gpu=2, GPTLstart_gpu succeeded through %d user starts\n", start_success);
  ret = GPTLcudadevsync ();
  if (handle_success == 1 && start_success == 1)
    return 0;
  else
    return -1;
}
