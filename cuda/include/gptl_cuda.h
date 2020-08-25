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
int GPTLcompute_chunksize (const int, const int);
int GPTLcudadevsync (void);
#pragma acc routine seq
__device__ int GPTLinit_handle_gpu (const char *, int *);
#pragma acc routine seq
__device__ int GPTLstart_gpu (const int);
#pragma acc routine seq
__device__ int GPTLstop_gpu (const int);
#pragma acc routine seq
__device__ void GPTLdummy_gpu (void);
#pragma acc routine seq
__device__ int GPTLmy_sleep (float);
#pragma acc routine seq
__device__ int GPTLget_wallclock_gpu (const int, double *, double *, double *);
#pragma acc routine seq
__device__ void GPTLwhoami (const char *);
#ifdef __cplusplus
};
#endif
#endif
