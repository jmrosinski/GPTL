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
// These first 3 are host routines
int GPTLget_gpu_props (int *, int *, int *, int *, int *, int *);
int GPTLcudadevsync (void);
__device__ int GPTLinit_handle_gpu (const char *, int *);
__device__ int GPTLstart_gpu (const int);
__device__ int GPTLstop_gpu (const int);
__device__ void GPTLdummy_gpu (void);
__device__ int GPTLmy_sleep (float);
__device__ int GPTLget_wallclock_gpu (const int, double *, double *, double *);
__device__ int GPTLget_warp_thread (int *, int *);
#ifdef __cplusplus
};
#endif
#endif
