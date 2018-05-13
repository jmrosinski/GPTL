typedef struct {
  long long start;
  long long stop;
} Timer;

__global__ void init_sim (Timer *, int);
__global__ void run_sim (void);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess) 
      {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
      }
  }
