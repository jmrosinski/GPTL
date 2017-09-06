/*
** f_wrappers.cu
**
** Author: Jim Rosinski
** 
** Fortran wrappers for timing library routines
*/
#include <stdlib.h>
#include <stdio.h>
#include "private.h" /* MAX_CHARS, private prototypes */
#include "gptl.h"    /* function prototypes */

#if ( defined FORTRANUNDERSCORE )

#define gptlstart_gpu gptlstart_gpu_
#define gptlinit_handle_gpu gptlinit_handle_gpu_
#define gptlstart_handle_gpu gptlstart_handle_gpu_
#define gptlstop_gpu gptlstop_gpu_
#define gptlstop_handle_gpu gptlstop_handle_gpu_
#define gptldummy_gpu gptldummy_gpu_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define gptlstart_gpu gptlstart_gpu__
#define gptlinit_handle_gpu gptlinit_handle_gpu__
#define gptlstart_handle_gpu gptlstart_handle_gpu__
#define gptlstop_gpu_gpu gptlstop_gpu_gpu__
#define gptlstop_handle_gpu gptlstop_handle_gpu__
#define gptldummy_gpu gptldummy_gpu__

#endif

extern "C" {

/* Local function prototypes */
__device__ int gptlstart_gpu (char *, long long);
__device__ int gptlinit_handle_gpu (const char *, int *, long long);
__device__ int gptlstart_handle_gpu (const char *, int *, long long);
__device__ int gptlstop_gpu (const char *, long long);
__device__ int gptlstop_handle_gpu (const char *, const int *, long long);
/* Fortran wrapper functions start here */

//JR Cannot dimension local cname[nc] because nc is an input argument
//JR Maybe a C99-compliancy issue?
__device__ int gptlstart_gpu (char *name, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstart_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstart_gpu (cname);
}

__device__ int gptlinit_handle_gpu (const char *name, int *handle, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlinit_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLinit_handle_gpu (cname, handle);
}

__device__ int gptlstart_handle_gpu (const char *name, int *handle, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstart_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstart_handle_gpu (cname, handle);
}

__device__ int gptlstop_gpu (const char *name, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstop_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstop_gpu (cname);
}

__device__ int gptlstop_handle_gpu (const char *name, const int *handle, long long nc)
{
  char cname[MAX_CHARS+1];
  const char *thisfunc = "gptlstop_handle_gpu";

  if (nc > MAX_CHARS)
    return GPTLerror_1s2d ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  return GPTLstop_handle_gpu (cname, handle);
}

//JR routines below are only for testing

__device__ int gptldummy_gpu (void)
{
  printf ("entered gptldummy_gpu\n");
  return GPTLdummy_gpu ();
}

__device__ extern int sub2 (char *);

__device__ int sub2_ (char *, long long);

__device__ int sub2_ (char *name, long long nc)
{
  int ret;
  char cname[MAX_CHARS+1];
  const char *thisfunc = "sub2_";

  if (nc > MAX_CHARS) {
    printf ("%s: %d exceeds MAX_CHARS=%d\n", thisfunc, nc, MAX_CHARS);
    return -1;
  }

  for (int n = 0; n < nc; ++n)
    cname[n] = name[n];
  cname[nc] = '\0';
  printf ("%s passing to sub2: cname=%s\n", thisfunc, cname);
  ret = sub2 (cname);
  return ret;
}

}
