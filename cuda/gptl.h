/*
** $Id: gptl.h.template,v 1.3 2011-03-28 20:55:19 rosinski Exp $
**
** Author: Jim Rosinski
**
** GPTL header file to be included in user code
*/

#ifndef GPTL_H
#define GPTL_H

/*
** Function prototypes
*/

__device__ extern int GPTLinitialize_gpu (void);

extern "C" {

__device__ extern int GPTLstart (const char *);
__device__ extern int GPTLinit_handle (const char *, int *);
__device__ extern int GPTLstart_handle (const char *, int *);
__device__ extern int GPTLstop (const char *);
__device__ extern int GPTLstop_handle (const char *, int *);

__device__ extern int GPTLreset (void);
__device__ extern int GPTLfinalize (void);
__device__ extern int GPTLenable (void);
__device__ extern int GPTLdisable (void);

};

#endif
