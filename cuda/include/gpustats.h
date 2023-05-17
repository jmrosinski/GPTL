#ifndef GPUSTATS_H
#define GPUSTATS_H

#include "api.h" // def. of Gpustats

namespace gpustats {
  extern __device__ void init_gpustats (Gpustats *, int);
  extern __global__ void fill_all_gpustats (Gpustats *, int *, int *);
}
#endif
