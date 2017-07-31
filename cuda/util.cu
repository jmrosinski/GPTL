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

__device__ int GPTLerror (const char *fmt, const unsigned int arg1, const int arg2)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, arg1, arg2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const char *str2)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str1, str2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const char *str2, const char *str3)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str1, str2, str3);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const int arg)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str1, arg);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const char *str2, const int arg1)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str1, str2, arg1);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const int arg1, const int arg2)
{
  (void) printf ("GPTL error:");
  (void) printf (fmt, str1, arg1, arg2);
  if (num_errors == max_errors)
    (void) printf ("Truncating further error print now after %d msgs", num_errors);
  ++num_errors;
  return -1;
}

__device__ int GPTLerror (const char *fmt, const char *str1, const int arg, const char *str2)
{
  (void) printf ("GPTL error:");
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
__device__ void GPTLnote (const char *str)
{
  (void) printf ("GPTLnote: %s\n", str);
}

/*
** GPTLreset_errors: reset error state to no errors
**
*/
__device__ void GPTLreset_errors (void)
{
  num_errors = 0;
}

/*
** GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
**
*/
__device__ int GPTLnum_errors (void)
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
__device__ void *GPTLallocate (const int nbytes, const char *caller)
{
  void *ptr;
  char str[MAXSTR];

  if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
    printf (str, "GPTLallocate from %s: malloc failed for %d bytes\n", nbytes, caller);

  ++num_errors;
  return ptr;
}
