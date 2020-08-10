/*
** $Id: util.c,v 1.13 2010-01-01 01:34:07 rosinski Exp $
*/

#include <stdio.h>
#include <stdlib.h>

#include "private.h"

__device__ static const int max_errors = 100;     // max number of error print msgs
__device__ static volatile int num_errors = 0;    // number of times GPTLerror was called
__device__ static volatile int mutex = 0;         // critical section unscrambles printf output

extern "C" {
__device__ static void grab_mutex (void);
}

#define RELENQUISH_MUTEX (mutex = 0);

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

__device__ int GPTLerror_1s (const char *fmt, const char *str)
{
  if (num_errors < max_errors) {
    grab_mutex ();
    (void) printf ("GPTL error:");
    (void) printf (fmt, str);
    ++num_errors;
    if (num_errors == max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    RELENQUISH_MUTEX;
  }
  return -1;
}

__device__ int GPTLerror_2s (const char *fmt, const char *str1, const char *str2)
{
  if (num_errors < max_errors) {
    grab_mutex ();
    (void) printf ("GPTL error:");
    (void) printf (fmt, str1, str2);
    ++num_errors;
    if (num_errors == max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    RELENQUISH_MUTEX;
  }
  return -1;
}

__device__ int GPTLerror_1s1d (const char *fmt, const char *str1, const int arg)
{
  if (num_errors < max_errors) {
    grab_mutex ();
    (void) printf ("GPTL error:");
    (void) printf (fmt, str1, arg);
    ++num_errors;
    if (num_errors == max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    RELENQUISH_MUTEX;
  }
  return -1;
}

__device__ int GPTLerror_2s1d (const char *fmt, const char *str1, const char *str2, const int arg1)
{
  if (num_errors < max_errors) {
    grab_mutex ();
    (void) printf ("GPTL error:");
    (void) printf (fmt, str1, str2, arg1);
    ++num_errors;
    if (num_errors == max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    RELENQUISH_MUTEX;
  }
  return -1;
}

__device__ int GPTLerror_1s2d (const char *fmt, const char *str1, const int arg1, const int arg2)
{
  if (num_errors < max_errors) {
    grab_mutex ();
    (void) printf ("GPTL error:");
    (void) printf (fmt, str1, arg1, arg2);
    ++num_errors;
    if (num_errors == max_errors)
      (void) printf ("Truncating further error print now after %d msgs\n", num_errors);
    RELENQUISH_MUTEX;
  }
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

__device__ void grab_mutex ()
{
  bool isSet; 
  // Grab a critical section (in this case for printing)
  do {
    // If mutex is 0, grab by setting = 1
    // If mutex is 1, it stays 1 and isSet will be false
    isSet = atomicCAS ((int *) &mutex, 0, 1) == 0; 
  } while ( !isSet);
  return;  // mutex is grabbed
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

}
