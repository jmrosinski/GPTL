/*
** $Id: util.c,v 1.13 2010-01-01 01:34:07 rosinski Exp $
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "private.h"

static bool abort_on_error = false;  /* flag says to abort on any error */
static int max_errors = 10;          /* max number of error print msgs */
static int num_errors = 0;           /* number of times GPTLerror was called */

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

int GPTLerror (const char *fmt, ...)
{
  va_list args;
  
  va_start (args, fmt);
  
  if (fmt != NULL && num_errors < max_errors) {
#ifdef HAVE_VPRINTF
    (void) fprintf (stderr, "GPTL error:");
    (void) vfprintf (stderr, fmt, args);
#else
    (void) fprintf (stderr, "GPTLerror: no vfprintf: fmt is %s\n", fmt);
#endif
    if (num_errors == max_errors)
      (void) fprintf (stderr, "Truncating further error print now after %d msgs",
		      num_errors);
  }
  
  va_end (args);
  
  if (abort_on_error)
    exit (-1);

  ++num_errors;
  return (-1);
}

/*
** GPTLset_abort_on_error: User-visible routine to set abort_on_error flag
**
** Input arguments:
**   val: true (abort on error) or false (don't)
*/
void GPTLset_abort_on_error (bool val)
{
  abort_on_error = val;
}

/*
** GPTLreset_errors: reset error state to no errors
**
*/
void GPTLreset_errors (void)
{
  num_errors = 0;
}

/*
** GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
**
*/
int GPTLnum_errors (void)
{
  return num_errors;
}

/*
** GPTLallocate: wrapper utility for malloc
**
** Input arguments:
**   nbytes: size to allocate
**
** Return value: pointer to the new space (or NULL)
*/
void *GPTLallocate (const int nbytes)
{
  void *ptr;

  if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
    (void) GPTLerror ("GPTLallocate: malloc failed for %d bytes\n", nbytes);

  return ptr;
}
