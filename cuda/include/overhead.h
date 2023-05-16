namespace overhead {
  __global__ void get_overhead_gpu (float *,            // Getting my warp index
				    float *,            // start/stop pair
				    float *,            // Underlying timing routine
				    float *,            // misc start code
				    float *,            // misc stop code
				    float *,            // self_ohd
				    float *,            // parent_ohd
				    float *,            // my_strlen ohd
				    float *,            // STRMATCH ohd
				    int *);             // return code from __global__
  __global__ void get_memstats_gpu (float *, float *);
}
