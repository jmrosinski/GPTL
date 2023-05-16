namespace util {
  extern {
    __device__ void get_mutex (volatile int *);
    __device__ void free_mutex (volatile int *);
    __device__ int error_1s (const char *fmt, const char *str);
    __device__ int error_2s (const char *fmt, const char *str1, const char *str2);
    __device__ int error_1s1d (const char *fmt, const char *str1, const int arg);
    __device__ int error_2s1d (const char *fmt, const char *str1, const char *str2, const int arg1);
    __device__ int error_2s3d (const char *fmt, const char *str1, const char *str2,
			       const int arg1, const int arg2, const int arg3);
    __device__ int error_1s2d (const char *fmt, const char *str1, const int arg1, const int arg2);
    __device__ void note_gpu (const char *str);
    __device__ void reset_errors_gpu (void);
    __device__ int get_maxwarpid_timed (void);

    __global__ void get_maxwarpid_info (int *, int *);
    __global__ void reset_gpu (const int, int *);
  }
}
