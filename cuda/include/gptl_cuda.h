/*
** $Id: gptl_cuda.h.template,v 1.3 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** GPTL header file to be included in user code
*/

#ifndef GPTL_CUDA_H
#define GPTL_CUDA_H

/*
** User-visible function prototypes
*/
#ifdef __cplusplus
extern "C" {
#endif
__device__ int GPTLinit_handle_gpu (const char *, int *);
__device__ int GPTLstart_gpu (const int);
__device__ int GPTLstop_gpu (const int);
__device__ void GPTLdummy_gpu (void);
__device__ int GPTLmy_sleep (float);
__device__ int GPTLget_wallclock_gpu (const int, double *, double *, double *);
__device__ int GPTLget_warp_thread (int *, int *);
__device__ int GPTLsliced_up_how (const char *);
__device__ int GPTLcuProfilerStart (void);
__device__ int GPTLcuProfilerStop (void);
#ifdef __cplusplus
};
#endif
#endif
