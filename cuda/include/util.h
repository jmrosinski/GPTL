#ifndef GPTLUTIL_H
#define GPTLUTIL_H

namespace util {
  extern __device__ void get_mutex (volatile int *);
  extern __device__ void free_mutex (volatile int *);
  extern __device__ int error_1s (const char *fmt, const char *str);
  extern __device__ int error_2s (const char *fmt, const char *str1, const char *str2);
  extern __device__ int error_1s1d (const char *fmt, const char *str1, const int arg);
  extern __device__ int error_2s1d (const char *fmt, const char *str1, const char *str2,
				    const int arg1);
  extern __device__ int error_2s3d (const char *fmt, const char *str1, const char *str2,
				    const int arg1, const int arg2, const int arg3);
  extern __device__ int error_1s2d (const char *fmt, const char *str1, const int arg1,
				    const int arg2);
  extern __device__ void note_gpu (const char *str);
  extern __device__ void reset_errors_gpu (void);

  extern __global__ void get_maxwarpid_timed (int *);
  extern __global__ void get_maxwarpid_found (int *);
  extern __global__ void reset_gpu (const int, int *);
}
#endif
