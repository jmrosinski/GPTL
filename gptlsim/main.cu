#include <stdio.h>
#include <cuda.h>
#include "./proto.h"

int main ()
{
  Timer *table_cpu;
  size_t nbytes;  // number of bytes to allocate
  int gridsize, blocksize;
  int maxwarps;

  printf ("Enter gridsize\n");
  (void) scanf ("%d", &gridsize);

  printf ("Enter blocksize\n");
  (void) scanf ("%d", &blocksize);

  printf ("Enter maxwarps (gridsize*blocksize=%d)\n", gridsize*blocksize);
  (void) scanf ("%d", &maxwarps);

  nbytes = maxwarps * sizeof (Timer);
  gpuErrchk (cudaMalloc (&table_cpu, nbytes));

  init_sim <<<1,1>>> (table_cpu, maxwarps);
  cudaDeviceSynchronize ();
  
  run_sim <<<gridsize,blocksize>>> ();
  cudaDeviceSynchronize ();
  return 0;
}
