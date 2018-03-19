/*
** $Id: util.c,v 1.13 2010-01-01 01:34:07 rosinski Exp $
*/

#include <stdio.h>
#include <stdlib.h>

#include "private.h"

__device__ static int max_errors = 1; /* max number of error print msgs */
__device__ static int num_errors = 0; /* number of times GPTLerror was called */

#define MAXSTR 256

/*
** GPTLerror: error return routine to print a message and return a failure
** value.
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
**
** Return value: -1 (failure)
*/

extern "C" {

__device__ int GPTLerror_1u1d (const char *fmt, const unsigned int arg1, const int arg2)
{
  static const char *thisfunc = "GPTLerror_1u1d";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, arg1, arg2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_1s (const char *fmt, const char *str)
{
  static const char *thisfunc = "GPTLerror_1s";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_2s (const char *fmt, const char *str1, const char *str2)
{
  static const char *thisfunc = "GPTLerror_2s";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str1, str2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_1s1d (const char *fmt, const char *str1, const int arg)
{
  static const char *thisfunc = "GPTLerror_1s1d";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str1, arg);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_2s1d (const char *fmt, const char *str1, const char *str2, const int arg1)
{
  static const char *thisfunc = "GPTLerror_2s1d";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str1, str2, arg1);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_1s2d (const char *fmt, const char *str1, const int arg1, const int arg2)
{
  static const char *thisfunc = "GPTLerror_1s2d";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str1, arg1, arg2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror_1s1d1s (const char *fmt, const char *str1, const int arg, const char *str2)
{
  static const char *thisfunc = "GPTLerror_1s1d1s";

  (void) printf ("%s: GPTL error:", thisfunc);
  (void) printf (fmt, str1, arg, str2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

/*
** GPTLnote: print a note
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
*/
__device__ void GPTLnote_gpu (const char *str)
{
  (void) printf ("GPTLnote_gpu: %s\n", str);
}

/*
** GPTLreset_errors: reset error state to no errors
**
*/
__device__ void GPTLreset_errors_gpu (void)
{
  num_errors = 0;
}

/*
** GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
**
*/
__device__ int GPTLnum_errors_gpu (void)
{
  return num_errors;
}

/*
** GPTLallocate: wrapper utility for malloc
**
** Input arguments:
**   nbytes: size to alTlocate
**
** Return value: pointer to the new space (or NULL)
*/
__device__ void *GPTLallocate_gpu (const int nbytes, const char *caller)
{
  void *ptr;
  cudaError_t ret;

  if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
    printf ("GPTLallocate_gpu from %s: malloc failed for %d bytes\n", nbytes, caller);

  /*
  if ( nbytes <= 0)
    printf ("GPTLallocate_gpu from %s: cannot allocate %d bytes\n", caller, nbytes);

  ret = cudaMalloc (&ptr, nbytes);
  if ( ret != cudaSuccess) {
    printf ("GPTLallocate_gpu from %s: cudaMalloc failed for %d bytes\n", caller, nbytes);
    printf ("%s\n", cudaGetErrorString(ret));
  }
  */
  return ptr;
}

}
