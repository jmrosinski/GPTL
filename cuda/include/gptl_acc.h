/*
** $Id: gptl_cuda.h.template,v 1.3 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** GPTL header file to be included in user code
*/

#ifndef GPTL_CACC_H
#define GPTL_CACC_H

/*
** User-visible function prototypes
*/

#ifdef __cplusplus
extern "C" {
#endif
// First 2 are host routines
int GPTLget_gpu_props (int *, int *, int *, int *, int *, int *);
int GPTLcudadevsync (void);
#pragma acc routine seq
int GPTLinit_handle_gpu (const char *, int *);
#pragma acc routine seq
int GPTLstart_gpu (const int);
#pragma acc routine seq
int GPTLstop_gpu (const int);
#pragma acc routine seq
void GPTLdummy_gpu (void);
#pragma acc routine seq
int GPTLmy_sleep (float);
#pragma acc routine seq
int GPTLget_wallclock_gpu (const int, double *, double *, double *);
#pragma acc routine seq
int GPTLget_warp_thread (int *, int *);
#ifdef __cplusplus
};
#endif
#endif
