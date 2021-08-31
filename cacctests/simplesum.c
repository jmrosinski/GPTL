#include "../config.h"
#include <openacc.h>
#include <stdio.h>
#include "gptl.h"
#include "gptl_acc.h"

/*
** Implement simple gang, worker, vector ACC directives to add numbers.
** PACKAGE will be defined when built as part of GPTL, but code can also be built
** standalone which won't add these extra GPTL diagnostics.
*/
int main ()
{
  int SMcount = 5;
  int warps_per_sm = 4;
  int warpsize = 32;
  int ndev;
  acc_device_t devtyp;
  int totwarps;
  int sum;
  int ret;
  int n;
  int khz, devnum, cores_per_sm, cores_per_gpu;

  devtyp = acc_get_device_type ();
#ifdef PACKAGE
  ret = GPTLget_gpu_props (&khz, &warpsize, &devnum, &SMcount,
			   &cores_per_sm, &cores_per_gpu);
  warps_per_sm = cores_per_sm / warpsize;
  ret = GPTLinitialize ();
#else
  acc_init (devtyp);
#endif
  int ganglen = SMcount;
  int worklen = warps_per_sm;
  int veclen = warpsize;
  totwarps = ganglen*worklen*veclen / warpsize;
  int smarr[totwarps];
  
  printf ("devtyp = %d\n", (int) devtyp);
  ndev = acc_get_num_devices (devtyp);
  printf ("ndev=%d\n", ndev);

  sum = 0;
  for (n = 0; n < totwarps; ++n)
    smarr[n] = -1;
  
#pragma acc parallel reduction(+:sum) copy(sum,smarr) copyin(ganglen,worklen,veclen)
  {
#pragma acc loop gang
    for (int g = 0; g < ganglen; ++g) {
#pragma acc loop worker
      for (int w = 0; w < worklen; ++w) {
#pragma acc loop vector
	for (int t = 0; t < veclen; ++t) {
	  sum += 1;
#ifdef PACKAGE
	  (void) GPTLget_sm_thiswarp (smarr);
#endif
	}
      }
    }
  }
  printf ("sum should be %d got %d\n", ganglen*worklen*veclen, sum);
#ifdef PACKAGE
  for (int n = 0; n < totwarps; ++n) {
    if (smarr[n] != -1)
      printf ("warp=%d ran on SM=%d\n", n, smarr[n]);
  }
#endif
  acc_shutdown (devtyp);
}
