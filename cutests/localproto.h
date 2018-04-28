extern __host__ int persist (int, int, int, int, int);
extern __host__ int sleep1 (int, int);
extern __host__ int getval_int (const char *, const int);
extern __global__ void warmup (void);
extern __global__ void donothing (void);
extern __global__ void doalot (int, int, int, int, float *, float *);
extern __device__ float doalot_log (int, int);
extern __device__ float doalot_log_inner (int, int);
extern __device__ float doalot_sqrt (int, int);
extern __device__ double doalot_sqrt_double (int, int);
extern __global__ void sleep (float, int);

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif
