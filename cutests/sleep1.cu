#include <stdio.h>
#include <stdlib.h>
#include "../gptl.h"
#include "./localproto.h"

__host__ int sleep1 (int outerlooplen, int oversub)
{
  int blocksize, gridsize;
  int ret;
  int n, nn;
  int totalwork;
  float dt;  
  int chunksize;
  int nchunks;
  static const char *thisfunc = "onlysleep";

  chunksize = MIN (GPTLcompute_chunksize (oversub, 1), outerlooplen);
  nchunks = (outerlooplen + (chunksize-1)) / chunksize;
  printf ("outerlooplen=%d broken into %d kernels of chunksize=%d\n",
	  outerlooplen, nchunks, chunksize);

  printf ("%s: issuing cudaMalloc calls to hold results\n", thisfunc);

  n = 0;
  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    printf ("chunk=%d totalwork=%d\n", n, MIN (chunksize, outerlooplen - nn));
    ++n;
  }

  printf ("Sleeping 1 second on GPU...\n");
  ret = GPTLstart ("sleep1ongpu");
  for (nn = 0; nn < outerlooplen; nn += chunksize) {
    totalwork = MIN (chunksize, outerlooplen - nn);
    blocksize = MIN (GPTLcores_per_sm, totalwork);
    gridsize = (totalwork-1) / blocksize + 1;
    sleep <<<gridsize, blocksize>>> (1.f, outerlooplen);
    cudaDeviceSynchronize();
  }
  ret = GPTLstop ("sleep1ongpu");

  return 0;
}
