/*
** $Id: gptl_ompacc.h.template,v 1.3 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** GPTL header file to be included in user C OpenMP GPU code
*/

#ifndef GPTL_CACC_H
#define GPTL_CACC_H

/*
** User-visible function prototypes
*/

#ifdef __cplusplus
extern "C" {
#endif
#pragma omp declare target
int GPTLinit_handle_gpu (const char *, int *);
int GPTLstart_gpu (const int);
int GPTLstop_gpu (const int);
void GPTLdummy_gpu (void);
int GPTLmy_sleep (float);
int GPTLget_wallclock_gpu (const int, double *, double *, double *);
int GPTLget_warp_thread (int *, int *);
int GPTLsliced_up_how (const char *);
int GPTLget_sm_thiswarp (int []);
#pragma omp end declare target
#ifdef __cplusplus
};
#endif
#endif
